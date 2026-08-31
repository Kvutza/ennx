# Reading the Sarrafi README

The root README is a modern Sarrafi reading edition for ENNX. Its language is
Godwari (`gdx`), targeted at Jain mercantile speech around Nana and Sumerpur in
Pali district. The text is encoded with characters from the Unicode Mahajani
block because Unicode does not encode a separate Sarrafi block.

This is a corpus-backed editorial text, not a transcription of one speaker.
Falna is the nearest surveyed Godwari site to Nana in the available language
survey. A speaker from the Nana/Sumerpur Jain community should review future
prose changes before this text is treated as a community standard.

## Canonical source

The Devanagari below is the canonical language source. The Sarrafi README is a
rendering of this text; it is not the source from which pronunciation should be
guessed.

```text
# ENNX · इ एन एन एक्स

## बणावो

ENNX रो साधारण काम `./ennx` सूं चालावो। Buck2 इणरो मुख्य build system है। raw Buck2 अर Bazel command खाली debugging अर compatibility खातर चालावो।

    ./ennx dev

wheel बणावण अर परखण खातर:

    ./ennx wheel

`./ennx wheel` Buck2 सूं CPython 3.13 रो wheel बणावै अर verify करै। CPython 3.12 सूं 3.14 तक रा platform wheel GitHub Releases मांय मिलै; PyPI मांय कोनी। आपरै Python ABI अर platform रो wheel install करो।

CUDA wheel बणावण अर parity परखण खातर Buck2 targets अर GitHub Actions मांय रहै।

तैयार wheel release मांय upload करण खातर:

    gh release upload vX.Y.Z PATH/TO/ennx-*.whl

## परखो

    ./ennx dev
    ./ennx dev --full
    ./ennx ci

bदलिया file: `./ennx dev`

सगळो repo: `./ennx ci`

पूरी gate: `./ennx ci`

परख रा दरजा अर benchmark री कसौटी: `docs/testing.md`

T4 CUDA development: `docs/colab.md`

API री हद: `docs/api.md`

## बिगत

सगळा CLI हुकम:

    ./ennx --help

मुख्य Buck2 path अर platform री बिगत: `docs/buck2.md`

secondary Bazel compatibility path: `docs/bazel.md`

सर्राफी पढ़ण अर बोलण री मदद: `docs/sarrafi.md`
```

## How to speak it

Read the commands and English technical terms as code. Read the Godwari prose
with these broad values:

| Devanagari | Broad reading | Meaning |
| --- | --- | --- |
| बणावो | `baṇāvo` | make or build |
| परखो | `parakho` | test or check |
| रो / रा / री | `ro / rā / rī` | agreeing genitive forms |
| सूं | `sū̃` | from, with, or by means of |
| अर | `ar` | and |
| इणरो | `iṇro` | its, masculine singular |
| खाली | `khālī` | only |
| खातर | `khātar` | for |
| बणावण | `baṇāvaṇ` | making or building |
| बणावै | `baṇāvai` | makes or builds |
| मांय | `mā̃y` | in or on |
| मिलै | `milai` | is available |
| कोनी | `konī` | is not |
| आपरै | `āparai` | for your |
| बदलिया | `badaliyā` | changed |
| सगळो / सगळा | `sagaḷo / sagaḷā` | all or the whole |
| पूरी | `pūrī` | complete |
| दरजा | `darajā` | levels or tiers |
| कसौटी | `kasauṭī` | criterion or test |
| हद | `had` | boundary |
| बिगत | `bigat` | details |
| हुकम | `hukam` | commands |
| सर्राफी | `sarrāfī` | Sarrafi |
| पढ़ण अर बोलण री मदद | `paṛhaṇ ar bolaṇ rī madad` | help with reading and speaking |

### Continuous reading

Speak the prose in this order; read the intervening commands literally:

```text
ENNX ro sādhāraṇ kām ./ennx sū̃ chālāvo.
Buck2 iṇro mukhya build system hai.
raw Buck2 ar Bazel command khālī debugging ar compatibility khātar chālāvo.

wheel baṇāvaṇ ar parakhaṇ khātar.

./ennx wheel Buck2 sū̃ CPython 3.13 ro wheel baṇāvai.
CPython 3.12 sū̃ 3.14 tak rā platform wheel GitHub Releases mā̃y milai;
PyPI mā̃y konī.
āparai Python ABI ar platform ro wheel install karo.

CUDA wheel baṇāvaṇ ar parity parakhaṇ khātar.
taiyār CUDA wheel release mā̃y upload karaṇ khātar.

parakho.
badaliyā file ar native Buck2 graph: ./ennx dev.
sagaḷo repo ar wheel: ./ennx dev --full.
pūrī gate: ./ennx ci.
parakh rā darajā ar benchmark rī kasauṭī: docs/testing.md.
T4 CUDA development: docs/colab.md.
API rī had: docs/api.md.

bigat.
sagaḷā CLI hukam: ./ennx --help.
mukhya Buck2 path ar platform rī bigat: docs/buck2.md.
secondary Bazel compatibility path: docs/bazel.md.
sarrāfī paṛhaṇ ar bolaṇ rī madad: docs/sarrafi.md.
```

Macrons mark long vowels. A tilde marks nasalization. `ṇ`, `ḷ`, `ṭ`, and `ṛ`
are retroflex; the tongue tip is curled back. These readings are broad and are
not claims about every speaker's narrow phonetic realization. In local speech,
unstressed `a` may approach `[ə]`, `v` may approach `[ʋ]`, and written `hai`
may surface near `[hɛ]`. Keep `sū̃` nasal; do not add a final pronounced `n`.

## Orthographic policy

- **Name:** ENNX calls the writing system Sarrafi, the Jain mercantile name
  relevant to this edition. "Mahajani" appears only when identifying Unicode's
  technical block name.
- **Vowels:** Sarrafi normally leaves medial vowels to morphological inference.
  This edition writes a following independent vowel letter where useful, as
  permitted by the encoded orthography, so technical prose is less ambiguous.
  Vowel length can still be ambiguous; the canonical Devanagari source and the
  broad reading above resolve it.
- **Clusters:** Consonants are written sequentially. There is no virama.
- **Nasalization:** `NA` may explicitly record nasalization, as in `sū̃` and
  `mā̃y`.
- **YA:** The encoded repertoire has no separate common `YA`; `JA` represents
  it where required.
- **Spacing:** Historical account writing did not enforce word spaces. This
  edition keeps spaces and modern punctuation for software documentation.
- **Technical vocabulary:** Commands, paths, product names, and API terms stay
  in Latin script. They are not assigned invented Godwari inflections.

## Evidence

- Anshuman Pandey, *Proposal to Encode the Mahajani Script in ISO/IEC 10646*,
  defines the encoded repertoire, vowel omission, optional explicit vowel
  letters, sequential clusters, nasalization, and `JA` for `YA`:
  <https://www.unicode.org/L2/L2011/11274-n4126-mahajani.pdf>.
- Abraham et al., *A Sociolinguistic Survey of the Marwari Language*, records
  Godwari data at Falna and Kherwa in Pali district. The Falna list supports the
  local sound system and forms including `bɦaɭɔ` "look", `hunɔ` "listen",
  `bolɔ` "speak", `gɔm` "village", and `kuɳə` "who":
  <https://www.sil.org/resources/archives/50815>.
- Glottolog identifies Godwari as ISO 639-3 `gdx`, Glottocode `godw1241`:
  <https://glottolog.org/resource/languoid/id/godw1241>.

The survey is a lexical and sociolinguistic source, not a complete grammar.
Forms not directly present in its Falna list use conservative Western
Rajasthani morphology. That distinction should remain explicit during review.
