import numpy as np
import re
import pandas as pd
try:
    import cudf
    _CUDF = True
except ImportError:
    cudf = None
    _CUDF = False

import unicodedata
from tqdm import tqdm 
from rapidfuzz import fuzz, process
from phonetic_fr import phonetic
from Levenshtein import distance as levenshtein_distance

try:
    from postal.expand import expand_address as _postal_expand
    LIBPOSTAL_AVAILABLE = True
except ImportError:
    LIBPOSTAL_AVAILABLE = False

def apply_ortho_correction_and_realign(df):
    is_cudf = type(df).__module__.startswith('cudf')
    df_pd = df.to_pandas() if is_cudf else df.copy()

    # Initialize adr_init_corr as a copy of adr_init
    df_pd['adr_init_corr'] = df_pd['adr_init']

    mask = (df_pd['ortho_pairs'] != '')
    print(f"rows to realign: {mask.sum()}")

    for i in tqdm(df_pd.loc[mask].index):
        pairs = {
            p.split('->')[0]: p.split('->')[1]
            for p in df_pd.loc[i, 'ortho_pairs'].split('|')
            if '->' in p
        }

        # Correct adr_init_corr only — adr_init untouched
        adr_init_corrected = df_pd.loc[i, 'adr_init']
        for brute_tok, geo_tok in pairs.items():
            adr_init_corrected = re.sub(
                rf'\b{brute_tok}\b', geo_tok, adr_init_corrected
            )
        df_pd.loc[i, 'adr_init_corr'] = adr_init_corrected

        # Rerun NW on corrected string
        char_score, token_score, brute_aligned, geo_aligned, unmatched_brute, unmatched_geo = \
            Needleman_Wunch_update(adr_init_corrected, df_pd.loc[i, 'adr_geo'], return_all=True)

        df_pd.loc[i, 'char_alignment_score']  = char_score
        df_pd.loc[i, 'token_alignment_score'] = token_score
        df_pd.loc[i, 'brute_aligned']         = brute_aligned
        df_pd.loc[i, 'geo_aligned']            = geo_aligned
        df_pd.loc[i, 'not_matched_brute']      = ' '.join(unmatched_brute) if unmatched_brute else ''
        df_pd.loc[i, 'not_matched_geo']        = ' '.join(unmatched_geo) if unmatched_geo else ''

    return df_pd


def both_sides_have_name_tokens(df_pd):
    mask_has_both = (df_pd['not_matched_brute'] != '') & (df_pd['not_matched_geo'] != '')
    mask_has_pos  = df_pd['pos_street_type'].notna()
    candidate     = mask_has_both & mask_has_pos

    result = pd.Series(False, index=df_pd.index, dtype=bool)
    if candidate.sum() == 0:
        return result

    sub = df_pd[candidate].copy()

    # Build the street name part of adr_init (everything after pos_street_type)
    # using vectorized string split and join
    sub['pos_int'] = sub['pos_street_type'].astype(int)
    sub['adr_init_name'] = sub.apply(
        lambda row: ' '.join(row['adr_init'].split()[row['pos_int']:])
        if pd.notna(row['adr_init']) else '', axis=1
    )

    # Check if any not_matched_brute token appears in the name part
    # using vectorized str.contains with word boundary
    def any_token_in_name(row):
        for tok in row['not_matched_brute'].split():
            if re.search(rf'\b{re.escape(tok)}\b', row['adr_init_name']):
                return True
        return False

    result.loc[candidate] = sub.apply(any_token_in_name, axis=1).astype(bool).values
    return result
    
def vectorized_is_phonetic_match(df_pd, candidate_mask):
    """
    Vectorized phonetic check — only runs on candidate rows.
    For each row, checks if any not_matched_brute token (len>=5)
    is a phonetic variant of any adr_geo token (len>=5).
    """
    result = pd.Series(False, index=df_pd.index, dtype=bool)
    if candidate_mask.sum() == 0:
        return result

    sub = df_pd[candidate_mask].copy()

    # Pre-compute phonetic codes for not_matched_brute tokens
    # and adr_geo tokens — avoid recomputing per row
    def compute_phonetics(token_str):
        tokens = [t for t in token_str.split() if t not in STOP_WORDS]
        return [(t, phonetic(t)) for t in tokens]

    sub['brute_phonetics'] = sub['not_matched_brute'].apply(compute_phonetics)

    def compute_geo_phonetics(row):
        adr_init_set = set(row['adr_init'].split()) if pd.notna(row['adr_init']) else set()
        tokens = [t for t in row['adr_geo'].split() 
                  if t not in adr_init_set and t not in STOP_WORDS]
        return [(t, phonetic(t)) for t in tokens]
    
    sub['geo_phonetics'] = sub.apply(compute_geo_phonetics, axis=1)    
    def check_phonetic_match(row):
        for _, pb in row['brute_phonetics']:
            for _, pg in row['geo_phonetics']:
                if pb == pg:
                    return True
                if levenshtein_distance(pb, pg) <= 2:
                    return True
        return False

    result.loc[candidate_mask] = sub.apply(check_phonetic_match, axis=1).astype(bool).values
    return result
    
STOP_WORDS = {'DE', 'DU', 'DES', 'LA', 'LE', 'LES', 'L', 'D', 'AU', 'AUX', 
              'EN', 'ET', 'A', 'UN', 'UNE', 'SUR', 'SOUS', 'PAR', 'POUR'}

def is_phonetic_match_row(row):
    """
    Returns True if any unmatched_brute token is a phonetic variant
    of any adr_geo token — protects ortho variants from geocoding error flag.
    """
    unmatched_b    = row['not_matched_brute'].split() if row['not_matched_brute'] != '' else []
    # adr_geo_tokens = row['adr_geo'].split() if pd.notna(row['adr_geo']) else []
    adr_geo_tokens = row['not_matched_geo'].split() if row['not_matched_geo'] != '' else []

    for tok_b in unmatched_b:
        if tok_b in STOP_WORDS:
            continue
        for tok_g in adr_geo_tokens:
            if tok_b in STOP_WORDS:
                continue
            if _is_phonetic_variant(tok_b, tok_g):
                return True
    return False


def compute_phonetics(token_str):
    tokens = [t for t in token_str.split() if t not in STOP_WORDS]
    return [(t, phonetic(t)) for t in tokens]

def _is_phonetic_variant(a, b, max_phonetic_distance=2, min_fuzzy_ratio=70):
    fa, fb = phonetic(a), phonetic(b)
    
    # guard against empty phonetic codes
    if not fa or not fb:
        return False
    
    fa_stripped = fa[:-1] if len(fa) > 2 else fa
    fb_stripped = fb[:-1] if len(fb) > 2 else fb
    
    # phonetic substring — only if substring covers > 65% of longer phonetic code
    longer = max(len(fa), len(fb))
    if fa_stripped in fb and len(fa_stripped) / len(fb) > 0.65:
        return True
    if fb_stripped in fa and len(fb_stripped) / len(fa) > 0.65:
        return True
    if fa in fb_stripped and len(fa) / len(fb_stripped) > 0.65:
        return True
    if fb in fa_stripped and len(fb) / len(fa_stripped) > 0.65:
        return True

    if fuzz.ratio(a, b) < min_fuzzy_ratio:
        return False
    if fa == fb:
        return True
    return levenshtein_distance(fa, fb) <= max_phonetic_distance
def _is_corrupted_stopword(tok, not_matched_geo_tokens):
    """
    Identifies corrupted French prepositions like DELA, DUI, DUN, DDE.
    Rule: 2-4 chars, starts with D, contains E or U, 
    not present in not_matched_geo (i.e. not a real street token).
    """
    if not (2 <= len(tok) <= 4):
        return False
    if not tok.startswith('D'):
        return False
    if not any(c in tok for c in ['E', 'U']):
        return False
    if tok in not_matched_geo_tokens:
        return False
    return True

def _is_mutual_abbreviation(tok_b, tok_g, char_score, max_len=5, min_char_score=85):
    """
    Detects cases where both sides have short tokens at the same position
    that are likely different abbreviations of the same word.
    Condition: both short + high overall alignment (rest of address matches well)
    """
    if len(tok_b) > max_len or len(tok_g) > max_len:
        return False
    if char_score < min_char_score:
        return False
    # Share at least first character
    if tok_b[0] != tok_g[0]:
        return False
    return True
    
def classify_unmatched_tokens_fast(row):
    unmatched_b     = row['not_matched_brute'].split() \
                      if row['not_matched_brute'] != '' else []
    adr_geo_tokens  = row['adr_geo'].split() if pd.notna(row['adr_geo']) else []
    adr_init_tokens = row['adr_init'].split() if pd.notna(row['adr_init']) else []
    pos             = row.get('pos_street_type')
    char_score      = row.get('char_alignment_score', 100)

    adr_init_set = set(adr_init_tokens)
    candidate_geo_tokens = [t for t in adr_geo_tokens if t not in adr_init_set]

    if not unmatched_b or not candidate_geo_tokens:
        return pd.Series({
            'ortho_pairs':       '',
            'geocoding_error':   False,
            'not_matched_brute': ' '.join(unmatched_b)
        })

    ortho_pairs     = []
    geocoding_error = False
    genuine         = []

    for tok_b in unmatched_b:
        tok_pos = adr_init_tokens.index(tok_b) if tok_b in adr_init_tokens else None

        if pd.notna(pos) and tok_pos is not None and tok_pos < pos:
            genuine.append(tok_b)
            continue

        match = process.extractOne(tok_b, candidate_geo_tokens,
                                   scorer=fuzz.ratio,
                                   score_cutoff=0)
        if match is None:
            genuine.append(tok_b)
            continue

        best_tok, best_ratio, _ = match

        is_phonetic = _is_phonetic_variant(tok_b, best_tok) if tok_b not in STOP_WORDS else False

        if best_ratio >= 83 and char_score >= 80:
            ortho_pairs.append(f"{tok_b}->{best_tok}")
        elif is_phonetic:
            ortho_pairs.append(f"{tok_b}->{best_tok}")
        elif best_ratio >= 83 and char_score < 80:
            genuine.append(tok_b)

        else:
            if (pd.notna(pos) and tok_pos is not None and tok_pos > pos - 1
                    and not is_phonetic
                    and tok_b not in STOP_WORDS):
                # BEFORE — was just: genuine.append(tok_b)
                # AFTER — check mutual abbreviation first
                if _is_mutual_abbreviation(tok_b, best_tok, char_score):
                    ortho_pairs.append(f"{tok_b}->{best_tok}")
                else:
                    genuine.append(tok_b)
            elif _is_corrupted_stopword(tok_b, set(row['not_matched_geo'].split())):
                pass
            elif _is_mutual_abbreviation(tok_b, best_tok, char_score):
                ortho_pairs.append(f"{tok_b}->{best_tok}")
            else:
                genuine.append(tok_b)

    return pd.Series({
        'ortho_pairs':       '|'.join(ortho_pairs),
        'not_matched_brute': ' '.join(genuine)
    })
def apply_unmatched_classification(df):
    is_cudf = type(df).__module__.startswith('cudf')
    df_pd = df.to_pandas() if is_cudf else df.copy()

    df_pd['not_matched_brute'] = df_pd['not_matched_brute'].fillna('')
    df_pd['not_matched_geo']   = df_pd['not_matched_geo'].fillna('')
    df_pd['ortho_pairs']       = ''

    # Pre-filter 1 — must have something in not_matched_brute
    mask = df_pd['not_matched_brute'] != ''

    # Pre-filter 2 — at least one long token in not_matched_brute
    # has a plausible length match in adr_geo 
    def is_ortho_candidate(row, min_len=4):
        brute_tokens = [t for t in row['not_matched_brute'].split() if t not in STOP_WORDS]
        if not brute_tokens:
            return False
        # SAME logic as classify_unmatched_tokens_fast — exclude already matched tokens
        adr_init_set = set(row['adr_init'].split()) if pd.notna(row['adr_init']) else set()
        geo_tokens   = [t for t in row['adr_geo'].split() 
                        if pd.notna(row['adr_geo']) and t not in adr_init_set and t not in STOP_WORDS]
        if not geo_tokens:
            return False
        return any(
            abs(len(bt) - len(gt)) <= 2
            or bt in gt
            or gt in bt
            for bt in brute_tokens
            for gt in geo_tokens
        )

    mask = mask & df_pd[mask].apply(is_ortho_candidate, axis=1)
    print(f"rows to classify: {mask.sum()} / {len(df_pd)}")
    
    if 'geocoding_error' not in df_pd.columns:
        df_pd['geocoding_error'] = False
    
    result = df_pd.loc[mask].apply(classify_unmatched_tokens_fast, axis=1)
    df_pd.loc[mask, ['ortho_pairs', 'not_matched_brute']] = result[['ortho_pairs', 'not_matched_brute']]
    # newly_flagged = result['geocoding_error'] == True
    # df_pd.loc[mask & newly_flagged.reindex(df_pd.index, fill_value=False), 'geocoding_error'] = True

    return df_pd
    
def normalize_address_series(dataframe, column, language='fr'):
    """
    Replaces normalisation_adresse(). Apply symmetrically to both
    adr_init and adr_geo — neither side is normalized toward the other.
    
    Pipeline per address:
        1. libpostal expand  (DR→DOCTEUR, ST→SAINT, BD→BOULEVARD, etc.)
        2. unicode cleanup   (accents, punctuation, uppercase)
        3. residual map      (corpus-specific contractions libpostal misses)
    """
    # Known libpostal over-expansions on French address corpora
    # key = what libpostal produces, value = what it should stay as
    LIBPOSTAL_REVERT = {
        'ARCADE': 'ARC',
        'PERIPHERIQUE':'PERI',
    }
    
    def _normalize_single(address, language='fr'):
        if not isinstance(address, str) or not address.strip():
            return address
    
        # Step 1 — libpostal
        if LIBPOSTAL_AVAILABLE:
            expansions = _postal_expand(address, languages=[language])
            text = expansions[0].upper() if expansions else address.upper()
        else:
            text = address.upper()
    
        # Step 2 — unicode normalization
        nfd = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().upper()
    
        # Step 3 — revert libpostal over-expansions
        # Only revert if the original address did NOT already contain the 
        # expanded form — i.e. libpostal introduced it, the user didn't write it
        original_upper = address.upper()
        tokens = text.split()
        corrected = []
        for tok in tokens:
            if tok in LIBPOSTAL_REVERT and LIBPOSTAL_REVERT[tok] not in original_upper.split():
                corrected.append(LIBPOSTAL_REVERT[tok])
            else:
                corrected.append(tok)
        text = ' '.join(corrected)
    
        # # Step 4 — residual map
        # for pattern, replacement in RESIDUAL_MAP.items():
        #     text = re.sub(pattern, replacement, text)
    
        return re.sub(r'\s+', ' ', text).strip()

    df = dataframe.copy()
    is_cudf = type(df).__module__.startswith('cudf')
    series_pd = df[column].to_pandas() if is_cudf else df[column]
    normalized = series_pd.apply(lambda x: _normalize_single(x) if pd.notna(x) else x)
    df[column] = cudf.Series(normalized) if is_cudf else normalized
    return df
def NWmatrix2strings(S,T,D):
    """
    Needleman-Wunch matrix to aligned strings

    Args:
        S (str): first string
        T (str): second string
        D (np.array or list of lists): Decision matrix

    Returns:
        S_aligned (str): first string aligned
        T_aligned (str): second string aligned
    """
    S_WITH_GAP = -1
    T_WITH_GAP = 1
    S_T_ALIGN = 0
    i = D.shape[0]-1
    j = D.shape[1]-1
    S_aligned = ""
    T_aligned = ""
    
    while i > 0 or j > 0:
        if D[i,j] == S_WITH_GAP:
            S_aligned = S[i-1]+S_aligned
            T_aligned = "-"+T_aligned
            i -= 1
        elif D[i,j] == T_WITH_GAP:
            T_aligned = T[j-1]+ T_aligned
            S_aligned = "-"+S_aligned
            j -= 1
        elif D[i,j] == S_T_ALIGN:
            S_aligned = S[i-1] + S_aligned
            T_aligned = T[j-1] + T_aligned
            j -= 1
            i -= 1
    return(S_aligned,T_aligned)
           
    
def Needleman_Wunch_update(S,T,return_all=False, return_unmatched=True):
    """
    Algorithme de Needleman-Wunsch pour calculer la matrice de score et de décision.

    Args:
        S (str): premier string
        T (str): second string
        return_matrix (bool): indique si la fonction doit retourner les matrices F et D.

    Returns:
        F (np.array): Matrice de score
        D (np.array): Matrice de décision
        alignment_score (int): Score global d'alignement (coin inférieur droit de F)
    """
    S = S.upper()    
    T = T.upper()
    
    # Définition des score
    T_WITH_GAP = 1
    S_WITH_GAP = -1
    S_T_ALIGN = 0
    decision = [T_WITH_GAP,S_WITH_GAP,S_T_ALIGN]
    
    gap_score = -1
    substitution_score = -10
    
    # Initialisation des matrices F (score) et D (décision)
    F = np.zeros((len(S)+1,len(T)+1))
    F[0,:] = range(len(T)+1)
    F[:,0] = range(len(S)+1)
    F *= gap_score ## gap score 
    D = np.zeros((len(S)+1,len(T)+1))
    D[0,1:] = T_WITH_GAP 
    D[1:,0] = S_WITH_GAP 
    
    for i in range(1,len(S)+1):
        for j in range(1,len(T)+1):
            comparison = 1 if (S[i-1]==T[j-1]) else substitution_score # = int(S[i-1]==T[j-1])*2-1 # * 10 
            options = [F[i,j-1] + gap_score , F[i-1,j] + gap_score , F[i-1,j-1] + comparison]
            F[i,j] = options[0]
            D[i,j] = T_WITH_GAP
            for o,val in enumerate(options):
                if val > F[i,j]:
                    F[i,j] = val
                    D[i,j] = decision[o]
                    
    ### Récupération des châines alignées 
                    
    S_aligned, T_aligned = NWmatrix2strings(S, T, D)

    unmatched_S, unmatched_T = get_unmatched_segments(S,T,S_aligned, T_aligned)


    # Calcul du score d'alignement par caractères
    matched_chars = sum(1 for s_char, t_char in zip(S_aligned, T_aligned) if s_char == t_char and s_char != '-')
    total_chars = max(len(S), len(T))
    char_alignment_score = matched_chars / total_chars * 100  # Pourcentage de caractères alignés

    # Calcul du score d'alignement par tokens (mots)

    matched_tokens = [token for token in S_aligned.split(' ') if token not in unmatched_S]
    if '' in matched_tokens : 
        matched_tokens.remove('')
    total_tokens = max(len(S_aligned.split(' ')), len(T_aligned.split(' ')))
    token_alignment_score = len(matched_tokens) / total_tokens * 100  # Pourcentage de tokens alignés
    
    if return_all : 
        return char_alignment_score, token_alignment_score, S_aligned, T_aligned, unmatched_S, unmatched_T
    
    if return_unmatched :
        return char_alignment_score, token_alignment_score, unmatched_S, unmatched_T

    
def extract_words_from_chars(chars):
    words = []
    current_word = []

    for char in chars:
        if char != ' ':  # Ignore les espaces
            current_word.append(char)
        else:
            # Si on rencontre un espace, ajoute le mot en cours (s'il y en a un) et réinitialise
            if current_word:
                words.append(''.join(current_word))
                current_word = []

    # Ajoute le dernier mot si la liste ne se termine pas par un espace
    if current_word:
        words.append(''.join(current_word))

    return words


def get_word_at_position(text, position):
    
    if position == -1 or pd.isna(text):
        return ""
    
    words = text.split()  # Divise la chaîne en une liste de mots
    current_pos = 0  # Position courante pour suivre l'index des lettres

    for word in words:
        word_length = len(word)
        
        # Vérifie si la position donnée se trouve dans le mot actuel
        if current_pos <= position < (current_pos + word_length):
            return word
        current_pos += word_length + 1  # Passe au mot suivant en tenant compte de l'espace

    return ""

def split_to_words(segment):
# Utiliser une expression régulière pour séparer en mots tout en gardant la ponctuation
    return [word.strip() for word in re.findall(r'\b\w+\b', segment)]


def get_unmatched_segments(S,T,S_aligned, T_aligned):
    """
    Retourne les segments non appariés entre deux strings alignés.

    Args:
        S_aligned (str): premier string aligné
        T_aligned (str): second string aligné

    Returns:
        unmatched_S (list): segments non appariés de S
        unmatched_T (list): segments non appariés de T
    """
    
    S = S.upper()
    T = T.upper()
    
    S_words = S.split(' ')
    T_words = T.split(' ')  
    

    
    unmatched_S = []
    unmatched_T = []
    
    matched_S = []
    
    current_unmatched_S = ""
    current_unmatched_T = ""
    
    filtered_S_words = []

    for s_char, t_char in zip(S_aligned, T_aligned):
        if s_char == '-' and t_char != '-':
            # Gap dans S, donc caractère non apparié dans T
            current_unmatched_T += t_char
            if current_unmatched_S:
                unmatched_S.append(current_unmatched_S)
                current_unmatched_S = ""
        elif t_char == '-' and s_char != '-':
            # Gap dans T, donc caractère non apparié dans S
            current_unmatched_S += s_char
            if current_unmatched_T:
                unmatched_T.append(current_unmatched_T)
                current_unmatched_T = ""
        else:
            matched_S.append(s_char)

            # Les deux caractères sont appariés
            if current_unmatched_S:
                unmatched_S.append(current_unmatched_S)
                current_unmatched_S = ""
            if current_unmatched_T:
                unmatched_T.append(current_unmatched_T)
                current_unmatched_T = ""

    # Ajouter les segments non appariés restants
    if current_unmatched_S:
        unmatched_S.append(current_unmatched_S)
    if current_unmatched_T:
        unmatched_T.append(current_unmatched_T)


    unmatched_S_words = []
    unmatched_T_words = []
    
    for segment in unmatched_S:
        unmatched_S_words.extend(split_to_words(segment))

    for segment in unmatched_T:
        unmatched_T_words.extend(split_to_words(segment))
        
    matched_S_cleaned = extract_words_from_chars(matched_S)

    filtered_matched = [word for word in matched_S_cleaned if (word in S_words) or (word in T_words)]
    filtered_matched_uncleaned = [word for word in matched_S_cleaned if word not in filtered_matched]
    
    filtered_S_words = [get_word_at_position(S, S.find(word)) for word in unmatched_S_words if S.find(word) != -1]
    filtered_T_words = [get_word_at_position(T, T.find(word)) for word in unmatched_T_words if T.find(word) != -1]
    for i in filtered_matched_uncleaned : 
        position = S.find(i)
        if position != -1:
            cleaned_word = get_word_at_position(S, position)
            if cleaned_word not in filtered_S_words : 
                filtered_S_words.append(cleaned_word)


    return list(dict.fromkeys(filtered_S_words)), list(dict.fromkeys(filtered_T_words))

def calculate_alignment_metrics(S_aligned, T_aligned):
    """
    Calcule les pourcentages et le score global d'alignement en fonction des identités,
    substitutions, et gaps dans l'alignement final.

    Args:
        S_aligned (str): premier string aligné
        T_aligned (str): second string aligné

    Returns:
        dict: Contient % identité, % gaps, % substitutions, et le score d'alignement
    """

    identity_score = 3
    substitution_score = -1
    gap_score = -2
    
    identities = substitutions = gaps = 0
    alignment_length = len(S_aligned)

    for s_char, t_char in zip(S_aligned, T_aligned):
        if s_char == t_char and s_char != "-":
            identities += 1
        elif s_char == "-" or t_char == "-":
            gaps += 1
        else:
            substitutions += 1

    # Calcul des pourcentages
    percent_identity = (identities / alignment_length) * 100
    percent_gaps = (gaps / alignment_length) * 100
    percent_substitutions = (substitutions / alignment_length) * 100

    # Calcul du score d'alignement
    alignment_score = (identities * identity_score) + (substitutions * substitution_score) + (gaps * gap_score)

    return {
        "percent_identity": percent_identity,
        "percent_gaps": percent_gaps,
        "percent_substitutions": percent_substitutions,
        "alignment_score": alignment_score
    }

def remplacer_types_de_voies(df, colonne, df_odonyme):
    """
    Remplace les types de voies dans une colonne donnée d'un DataFrame, en utilisant 
    les synonymes et termes correspondants d'un autre DataFrame (df_odonyme).

    Args:
        df (pd.DataFrame): DataFrame contenant la colonne à modifier.
        colonne (str): Nom de la colonne où effectuer les remplacements.
        df_odonyme (pd.DataFrame): DataFrame contenant les colonnes 'synonym' et 'termn'.

    Returns:
        pd.DataFrame: DataFrame modifié avec les remplacements effectués.
    """
    dataframe = df.copy()

    # Trier df_odonyme par la longueur décroissante des synonymes
    df_odonyme = df_odonyme.assign(synonym_length=df_odonyme['synonym'].str.len())
    df_odonyme = df_odonyme.sort_values(by='synonym_length', ascending=False)

    # Effectuer les remplacements pour chaque synonyme dans l'ordre trié
    for _, row in df_odonyme.to_pandas().iterrows():
        synonym = row['synonym'].upper()
        replacement = row['terme'].upper() + " "

        # Utiliser str.replace pour les remplacements avec correspondance stricte
        dataframe[colonne] = dataframe[colonne].str.replace(
            rf'\b{synonym}\b', replacement, regex=True
        )
        
    return dataframe



def find_most_common_biaises(df,to_clean): 
    
    
    """
    3. Récupération des éléments non alignés de l'adresse brute et de l'adresse géocodée 
    4. Nettoyage des éléments : 
        - suppression des éléments nuls 
        - suppression des redondances 
        - séparation des numéros et mots 
        - suppression des numéros (-> idée : adresse avec num non alignée = err geocodage ?) 
    """
    not_matched_brute = pd.Series(df['not_matched_brute'].dropna())
    not_matched_geo =  pd.Series(df['not_matched_geo'].dropna())

    not_matched_brute_noNum = [' '.join(re.findall(r'\b[^\W\d_]+\b',x)) for x in not_matched_brute]
    not_matched_geo_noNum = [' '.join(re.findall(r'\b[^\W\d_]+\b',x)) for x in not_matched_geo]
    
    list_not_matched_brute_noNum = pd.Series([re.sub(r'[\d ]+', '', x) for x in not_matched_brute_noNum if x != ''])
    list_not_matched_geo_noNum = pd.Series([re.sub(r'[\d ]+', '', x) for x in not_matched_geo_noNum if x != ''])
    
    """5. Transformation de la liste des éléments alignées (string) vers des éléments individuel   
    - ensemble des éléments supplémentaires de l'adresse brute : {bruits}  
    - ensemble des éléments supplémentaires de l'adresse géocodée : {complement}
   """ 
    list_bruit_split = list_not_matched_brute_noNum.str.split(' ') 
    list_bruit_exploded = list_bruit_split.explode()
    bruit_count = pd.DataFrame(list_bruit_exploded.value_counts()).rename(columns={0: 'count'})
    
    list_compl_split = list_not_matched_geo_noNum.str.split(' ') 
    list_compl_exploded = list_compl_split.explode()
    compl_count = pd.DataFrame(list_compl_exploded.value_counts()).rename(columns={0: 'count'})
    compl_count = compl_count[compl_count['count']>=2]



    """ 6. Filtre des éléments non alignées de l'adresse brutes avec les compléments  
        {bruits filtres} = {bruits} minus {complement}
        
        SUPP DE CETTE PARTIE POUR AUTOMATISATION
        ##Probleme - filtre des éléments pertinents tel que "CHEZ" ou autre pouvant être considérés comme biais - retrait manuel 
    """
    common = bruit_count.index.intersection(compl_count.index)


    # unwanted_elements = ['HOPITAL', 'CHEZ', 'BATIMENT', 'LOTISSEMENT', 'APPT', 'PREMIER', 'ESC', 'GROUPE' , 'SECOURS', 'HOTEL' , 
    #                      'CO' ,'DOMICILE','APPARTEMENT','MAISON','ESCALIER' ]
    
    # common = common.difference(unwanted_elements)
    
    bruits_filtres = bruit_count.drop(common)
    
    """ 7. Filtre des éléments non alignées de l'adresse brutes avec les noms propres   
        {bruits vrais} = {bruits filtres} minus ( {prénoms} U {voies} ) """
    
    common = bruits_filtres.index.intersection(to_clean)
    bruits_vrais = bruits_filtres.drop(common)

    
    """8. Filtre des éléments non alignés de l'adresses brutes comportant des éléments de voiries, issus d'erreur de typographie"""
    pattern = r"\b(?:AVENUE|RUE|PLACE|IMPASSE|BOULEVARD|ALLEE|SQUARE|ROUTE|ESPLANADE|CHEMIN|GRANDE\sRUE|ROND\sPOINT|FAUBOURG|JARDIN|VIA|GALERIE|VOIE|QUAI|PASSAGE|COUR|COURS|CITE|PARVIS|HAMEAU|VILLA|VILLAGE|VILLE|TERRASSE|PROMENADE|SENTIER|CARREFOUR|CHAUSSEE|DOMAINE|CLOS|MOULIN|CENTRE|MAIL|BOIS|PROMENEE|VALLEE|RESIDENCE|QUARTIER|LOTISSEMENT|TRAVERSE|LIEU\sDIT|FERME|LE\sBOURG|PARC|COTE)[A-Za-z]+"

    bruits_fin = bruits_vrais[~bruits_vrais.index.str.contains(pattern, regex=True)]
    
   
    return bruits_fin.sort_values('count',ascending=False)


def find_pos_elem(df, column,elem,col_elem,col_contains_elem,col_pos_elem,drop_col_elem=True):
    """
    dataframe containing a column in which to search the position of elements 
    
    column : column in which to find position of elements 
    elem : Series of element to search 
    col_elem : column in which to stock the corresponding element found
    col_contains_elem : boolean True if one of the element of elem is present in the column 
    col_pos_elem : position of the element in the string 

    """
    dataframe = df.copy()
    elem_pd = elem.to_pandas() if hasattr(elem, 'to_pandas') else elem
    pattern_voies = r'\b(?:' + '|'.join(re.escape(word) for word in elem_pd) + r')\b'
    dataframe[col_contains_elem] = dataframe[column].str.contains(pattern_voies)
    
    dataframe[col_elem]=dataframe.loc[dataframe[col_contains_elem], column].str.extract(f'({pattern_voies})', expand=False)

    is_cudf = type(dataframe).__module__.startswith('cudf')
    if is_cudf:
        # cudf: str.split() + .list.index() work natively on list-typed columns
        # _tok_col added to dataframe so it can be dropped uniformly at the end
        _tok_col = '_tokens_tmp_'
        dataframe[_tok_col] = dataframe[column].str.split()
        dataframe[col_pos_elem] = dataframe[_tok_col].list.index(dataframe[col_elem])
        if drop_col_elem:
            dataframe = dataframe.drop([col_elem, _tok_col], axis=1)
        else:
            dataframe = dataframe.drop([_tok_col], axis=1)
    else:
        # pandas: tokens stay as a local variable — never written to the dataframe
        # so there is nothing extra to drop beyond col_elem
        tokens = dataframe[column].str.split()
        found  = dataframe[col_elem]
        def _pos(idx):
            toks = tokens.iloc[idx]
            val  = found.iloc[idx]
            if not isinstance(toks, list) or not isinstance(val, str):
                return np.nan
            try:
                return float(toks.index(val))
            except ValueError:
                return np.nan
        dataframe[col_pos_elem] = pd.Series(
            [_pos(i) for i in range(len(dataframe))],
            index=dataframe.index,
            dtype=float,
        )
        if drop_col_elem:
            dataframe = dataframe.drop([col_elem], axis=1)

    return dataframe 



def freq_couverture(dataframe,bruit,colonne):
    
    df = dataframe[(~dataframe[colonne].isna())&(dataframe[colonne]!="")]

    ##Serie contenant les bruits vrais contenus dans le dataframe : bruit_vrais_w_street 
    pattern_biaises = r'\b(?:' + '|'.join(re.escape(word) for word in bruit) + r')\b'

    ### Recherche des termes dans le dataframe
    df['biaises_found'] = df[colonne].str.findall(pattern_biaises)

    bruit_trouves = df[['id', 'biaises_found']].explode('biaises_found').reset_index()
    
    bruit_trouves.rename(columns={'id':'row_id','biaises_found': 'biais'}, inplace=True)
    bruit_trouves = bruit_trouves.dropna(subset="biais")
    bruit_trouves = bruit_trouves.drop('index',axis=1)

    df_groupby = bruit_trouves.groupby('biais').agg(list)

    df_groupby['list_size'] = df_groupby['row_id'].apply(len)

    df_sorted_gp = df_groupby.sort_values(by='list_size', ascending=True).drop(columns=['list_size'])

    seen_biaises = set()
    cumsum_values = []

    for list_of_row_id in df_sorted_gp['row_id']:  # already pandas — no .to_pandas() needed
        # Add unique values to the seen set
        seen_biaises.update(list_of_row_id)
        # The cumulative count is the size of the seen set
        cumsum_values.append(len(seen_biaises))

    # Assign the corrected cumsum values to the DataFrame
    df_sorted_gp['cumsum'] = cumsum_values
    
    df['biaises_found'] = df['biaises_found'].astype(str)
    nb_row_w_biais = len(df[df['biaises_found']!="[]"])

    df_sorted_gp['freq_globale_cum'] = df_sorted_gp['cumsum']/ len(dataframe)
    df_sorted_gp['freq_biais_cum'] = df_sorted_gp['cumsum']/ nb_row_w_biais
    
    return df_sorted_gp

def filter_by_word_in_column(df, column_name, word):
    

    df_filtre = df[df[column_name].str.contains(rf'\b{word}\b', regex=True)]
    
    
    
    return df_filtre

def calculer_distance_euclidienne(x1, y1, x2, y2):
    try:
        # Compute Euclidean distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return np.round(distance,3)
    except Exception as e:
        print(f"Erreur lors du calcul de la distance : {e}")
        return np.nan


def replace_columns_from_filepath(filepath): 
    
    with fiona.open(filepath, encoding='latin-1') as src:
        schema = src.schema
        
    gdf = gpd.read_file(filepath,encoding='latin-1').set_crs(2154)
    range_col = np.arange(len(gdf.columns) -1)
    schema = list(schema['properties'].keys())

    dic_col = {}
    for i in range(len(gdf.columns)-1): 
        dic_col[range_col[i]] = schema[i]
    
    gdf = gdf.rename(dic_col,axis=1)
    
    return gdf 

def replace_columns_from_df(df,gdf): 
    
    columns = list(df.columns)
        
    range_col = np.arange(len(columns))

    dic_col = {}
    for i in range(len(columns)): 
        dic_col[range_col[i]] = columns[i]
    
    gdf = gdf.rename(dic_col,axis=1)
    
    return gdf 

def create_df_for_figure(df,df_iris,df_revenus) : 
    ## reference coordinates : x_L93_ref, y_L93_ref
    gdf_init = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x_L93_ref, df.y_L93_ref)).set_crs(epsg=2154)
    gdf_init = replace_columns_from_df(df,gdf_init).drop('Unnamed: 0',axis=1)
    df_join_iris = gpd.sjoin(gdf_init, df_iris[['CODE_IRIS','geometry']], how="left", predicate='within')
    df_join_iris.drop('index_right', axis=1, inplace=True)
    df_init = df_join_iris.rename({'CODE_IRIS':'CODE_IRIS_init'},axis=1)
    
    ## geocoded coordinates : longitude, latitude
    gdf_geo = (gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude)).set_crs(epsg=4326)).to_crs(epsg=2154)
    df_join_iris = gpd.sjoin(gdf_geo, df_iris[['CODE_IRIS','geometry']], how="left", predicate='within')
    df_join_iris.drop('index_right', axis=1, inplace=True)
    df_geo = df_join_iris.rename({'CODE_IRIS':'CODE_IRIS_geocoded'},axis=1)
    
    
    df_fin = pd.concat([df_init,df_geo['CODE_IRIS_geocoded']],axis=1)

    ## Join to get median income associated to initial IRIS code 
    df_fin = df_fin.dropna(subset="CODE_IRIS_geocoded")
    df_rev_geoc = pd.merge(df_fin,df_revenus, how="left",left_on="CODE_IRIS_geocoded",right_on="CODE_IRIS")
    # df_rev_geoc = cudf.merge(df_fin,df_revenus,how="left",left_on="CODE_IRIS_geocoded",right_on="CODE_IRIS")
    df_rev_geoc = df_rev_geoc.rename({'DISP_MED20':'DISP_MED20_geocoded'},axis=1)
    df_rev_geoc = df_rev_geoc.drop('CODE_IRIS',axis=1)
    
    ## Join to get median income associated to geocoded IRIS code 
    df_rev_init = df_rev_geoc.dropna(subset="CODE_IRIS_init")
    # df_rev_init = df_rev_init.merge(df_revenus,how="left",left_on="CODE_IRIS_init",right_on="CODE_IRIS")
    df_rev_init = pd.merge(df_rev_init,df_revenus,how="left",left_on="CODE_IRIS_init",right_on="CODE_IRIS")
    df_rev_init = df_rev_init.rename({'DISP_MED20':'DISP_MED20_init'},axis=1)
    
    df_rev = df_rev_init.copy()
    
    df_rev["DISP_MED20_geocoded"] = df_rev["DISP_MED20_geocoded"].astype(float)
    df_rev["DISP_MED20_init"] = df_rev["DISP_MED20_init"].astype(float)
    
    df_rev["y_WGS84_ref"] = df_rev["y_WGS84_ref"].astype(float)
    df_rev["x_WGS84_ref"] = df_rev["x_WGS84_ref"].astype(float)
    
    df_rev["latitude"] = df_rev["latitude"].astype(float)
    df_rev["longitude"] = df_rev["longitude"].astype(float)
    
    ## Distance between the two coordinates 
    df_rev["distance_km"] = calculer_distance_euclidienne(df_rev['y_WGS84_ref'].values, df_rev['x_WGS84_ref'].values, df_rev['latitude'].values, df_rev['longitude'].values)
    
    df_rev["distance_m"] = df_rev['distance_km']*1000
    
    ## Difference of median income between the two IRIS  
    df_rev["diff_revenu"] =pd.to_numeric(df_rev["DISP_MED20_init"]) - pd.to_numeric(df_rev["DISP_MED20_geocoded"])

    return df_rev