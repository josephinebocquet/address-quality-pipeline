# LICENSE

## Project code and reference data

Copyright (c) [2026] [Josephine Bocquet / APHP]

[Etalab Open Licence 2.0]

---

### Etalab Open Licence 2.0

If this pipeline and its reference data are produced in the context of a public
administration or a public health institution, the Etalab Open Licence 2.0 is
the legally designated licence for French public-sector software and data
(SPDX: `Etalab-2.0`, compatible with CC-BY and OGL).
See: https://www.etalab.gouv.fr/licence-ouverte-open-licence/

---

## Third-party attributions

This project uses the following external tools and datasets. Their licences
apply to their respective components and do not extend to this project's own
code or reference data.

### addok — French geocoding engine

- **Source:** https://github.com/addok/addok
- **Licence:** MIT
- **Role:** The geocoding engine used to produce the `result_*` columns
  (housenumber, name, postcode, city, latitude, longitude) from raw addresses.
  This pipeline analyses the output of the BAN API powered by addok; it does
  not redistribute or modify addok itself.

### API Adresse / Base Adresse Nationale (BAN)

- **Source:** https://adresse.data.gouv.fr
- **Licence:** Etalab Open Licence 2.0 (SPDX: `Etalab-2.0`)
- **Role:** The national French address reference dataset indexed by addok.
  The geocoded coordinates used as ground truth in this pipeline (`adr_geo`,
  `result_name`, `latitude`, `longitude`) are derived from BAN data.
- **Attribution:** © Base Adresse Nationale — Etalab / DINUM

### INSEE Filosofi — Revenus disponibles par IRIS

- **Source:** https://www.insee.fr/fr/statistiques/6036907
- **Licence:** Etalab Open Licence 2.0
- **Role:** Median disposable income by IRIS zone (`DISP_MED20`), used in
  step 04 to quantify the socio-economic impact of geocoding errors.
- **Attribution:** © INSEE — Base de données Filosofi 2020

### IGN CONTOURS-IRIS

- **Source:** https://geoservices.ign.fr/irisge
- **Licence:** Etalab Open Licence 2.0 (since 1 January 2021, all IGN public
  data is released under this licence)
- **Role:** IRIS administrative boundary polygons used for spatial joins in
  step 04.
- **Attribution:** © IGN — CONTOURS-IRIS®