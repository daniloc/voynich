# Attributions & third-party sources

This repository's own contents — the analysis code (`analysis/`), the written findings
and documentation (`FINDINGS.md`, `docs/`, `README.md`, the `INDEX.md` files), and the
derived data produced by that code — are covered by the repository `LICENSE`.

Everything below was produced by third parties and is included here for research use
under the terms noted. Where a source asks for attribution, this file is that
attribution; where a source is public domain, it is listed for provenance. If you
reuse any of these assets, honor the original terms, not this repository's license.

## Primary source — the manuscript

- **Voynich Manuscript** — Yale University, Beinecke Rare Book & Manuscript Library,
  **MS 408**. The manuscript is 15th-century and in the **public domain**. Digital
  images (`images/facsimile/`, the folio crops in `images/crops/`, and the IIIF fetches
  driven by `data/iiif/`) come from Yale's public IIIF service, provided under Yale's
  open-access policy for public-domain holdings. Faithful reproductions of a
  public-domain work carry no new copyright.

## Transcription

- **`data/corpus/LSI_ivtff.txt`** — the Landini–Stolfi Interlinear (LSI) transcription
  of the manuscript in EVA, derived from Gabriel Landini's `INTERLN.EVT`, edited by
  Jorge Stolfi, with the Takahashi transcription as the primary interlinear row, and a
  2021 alignment pass by Massimiliano Zattera. A community research resource,
  redistributed with its full credit header intact (see the top of the file). Source:
  <http://www.voynich.nu/data/> (René Zandbergen's site). Please preserve the header
  credits on any redistribution.

## Language control corpora (`data/controls/`)

Used only as statistical baselines (letter/entropy/morphology distributions); no
control text is reproduced in the findings.

- **`english.txt`, `finnish.txt`, `latin.txt`** — derived from **Project Gutenberg**
  public-domain texts. The works are public domain; "Project Gutenberg" is a trademark
  with its own terms — retain the Gutenberg license headers in these files, or strip all
  Gutenberg references, if you redistribute. <https://www.gutenberg.org>
- **`maori_bible.xml`** and the other Bible-derived corpora — 19th-century Bible
  translations (public domain), packaged via Christos Christodoulopoulos & Mark Steedman,
  *"A massively parallel corpus: the Bible in 100 languages"* (Lang. Resources &
  Evaluation, 2015). <https://github.com/christos-c/bible-corpus>
- **`_mri_tatoeba` / `_mri_tatoeba.bz2`** (Māori sentences) — from the **Tatoeba
  Project**, licensed **CC BY 2.0 FR**. Attribution: © Tatoeba contributors,
  <https://tatoeba.org>.
- **`hawaiian.txt`, `hungarian.txt`, `swahili.txt`, `turkish.txt`, `maori.txt`** —
  public-domain / openly-licensed text samples (Bible-translation and Gutenberg-class
  sources) used as CV-heavy and agglutinative baselines. See `data/INDEX.md` for the
  per-file note. (Re K2: the Hawaiian sample is a 19th-century Bible edition; its
  reduced diacritics are the source of the okina caveat logged in `FINDINGS.md`.)

## Historical abbreviation controls (`data/external/`)

- **Cappelli downloadable data** — University of Zurich, Ad fontes, derived
  from Adriano Cappelli's *Lexicon Abbreviaturarum* (2nd ed., 1928). Ad fontes
  states that the images and information are public domain and may be reused
  without restriction. The checked-in derivative retains Latin records whose
  stated period intersects the fourteenth or fifteenth century.
  <https://www.adfontes.uzh.ch/en/ressourcen/abkuerzungen/cappelli-daten-zum-download>
- **Nuremberg Letterbooks v1** — Martin Mayr et al., *Nuremberg
  Letterbooks: A Multi-Transcriptional Dataset of Early 15th Century
  Manuscripts for Document Analysis*, Letterbooks 2–5 (1408–1423), licensed
  **CC BY 4.0**. Dataset DOI: <https://doi.org/10.5281/zenodo.13881575>.
  Accompanying data paper: <https://doi.org/10.1038/s41597-025-05144-z>.

## Comparanda manuscripts (`images/comparanda/`)

Digitizations of other medieval manuscripts, fetched during the zodiac-exemplar hunt
(motive findings M9/M11/M13) and used here for scholarly visual comparison. The
manuscripts are all public-domain by age; the **digital images** are provided by the
holding institutions under the licenses below. This repository's use is non-commercial
research. If you redistribute the image files, comply with each institution's terms; the
cleanest path is to **not** redistribute the images and instead re-fetch from the IIIF
manifests in `data/iiif/` (which is how they were obtained).

| Directory | Manuscript | Holding institution | Image license |
|---|---|---|---|
| `cand_ljs449/` | LJS 449 | University of Pennsylvania (OPenn) | Public domain / CC0 |
| `cand_cpg291/` | Cod. Pal. germ. 291 | Universitätsbibliothek Heidelberg | CC BY-SA 4.0 — **attribution + share-alike** |
| `cand_clm13076/` | Clm 13076 | Bayerische Staatsbibliothek (BSB) München | CC BY-NC-SA 4.0 |
| `cand_cgm28/` | Cgm 28 | Bayerische Staatsbibliothek (BSB) München | CC BY-NC-SA 4.0 |
| `cand_md2/` | Md 2 (Tübinger Hausbuch) | Universitätsbibliothek Tübingen | CC BY-NC-SA 3.0 (verify per-image) |
| `csg827/` | Cod. Sang. 827 | Stiftsbibliothek St. Gallen (e-codices) | CC BY-NC 4.0 |
| `cgm312_*.jpg` (in `images/comparanda/`) | Cgm 312 | Bayerische Staatsbibliothek (BSB) München | CC BY-NC-SA 4.0 |

Additional catalog references consulted (text/facts only, no images redistributed):
Heidelberg Cod. Pal. germ. 291; BSB Cgm 312; the Schürstab codex (`_schurstab_manifest.json`);
Nürnberg Staatsarchiv Hs. 426 (catalog-attested, not digitized). Note: the legal force of
these institutional image licenses over faithful reproductions of public-domain works is
contested (cf. *Bridgeman v. Corel*; German UrhG §68); they are honored here regardless as
a courtesy and to keep the repository trouble-free.

## External vision models

- **DINOv2-S/14** — Meta AI Research's self-supervised vision-transformer
  backbone, used frozen to derive the K26 guarded page and K27 complete-object
  embeddings. Code and standard model weights are licensed
  **Apache License 2.0**. Official
  repository: <https://github.com/facebookresearch/dinov2>, pinned here at
  revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The standard
  `dinov2_vits14_pretrain.pth` weights are fetched from Meta's published URL
  and are not vendored; SHA-256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
  The checked-in JSON contains derived feature vectors, not model weights.
- **Surya layout2** — Datalab's public document-layout detector, used in K27
  to propose `Image`, `Figure`, `Diagram`, and related visual regions.
  Repository: <https://github.com/datalab-to/surya>. Model card:
  <https://huggingface.co/datalab-to/surya_layout2>. The weights are licensed
  **AI Pubs OpenRAIL-M**, pinned at revision
  `0aee81d5fd9275c0582e545bf3a56944b1e75679`, and are fetched rather than
  vendored.
- **HORAE detection models** — the fine-tuned YOLO12s medieval page-layout
  detector from the HORAE project, used as a secondary K27 decoration proposal
  source. Model release: <https://doi.org/10.5281/zenodo.17279775>. Training
  dataset: <https://doi.org/10.5281/zenodo.17279364>. The released model
  artifacts are licensed **CC BY 4.0** and are fetched rather than vendored.
- **SAM 2.1** — Meta's Segment Anything 2.1 tiny checkpoint, used in K27 to
  turn public detector boxes into spatial object guards. Code and official
  checkpoints are licensed **Apache License 2.0**. Repository:
  <https://github.com/facebookresearch/sam2>. The checkpoint is fetched rather
  than vendored.

## Scholarship cited (facts and findings, not reproduced text)

The motive-forensics series draws on published Voynich scholarship — René Zandbergen,
Lisa Fagin Davis, Nick Pelling, Claire Bowern & Luke Lindemann, Marcelo Montemurro &
Damián Zanette, Élisabeth Pellegrin, and others — for dates, provenance, and
paleographic facts. Specific attributions are inline in `FINDINGS.md` and `docs/RUNS.md`.
Facts are not copyrightable; no substantial passages of any of these works are reproduced.
