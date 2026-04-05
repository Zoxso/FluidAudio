---
title: 2007.10310v3
source_url: file:///Users/brandonweng/code/mobius/knowledge/models/2007.10310v3.pdf
retrieved_at: 2025-10-23T23:23:16Z
---
# CoVoST 2 and Massively Multilingual Speech-to-Text Translation

Changhan Wang*, Anne Wu*, Juan Pino*

Facebook AI
{changhan, annewu, juancarabina}@fb.com

## Abstract

Speech-to-text translation (ST) has recently become an increasingly popular topic of research, partly due to the development of benchmark datasets. Nevertheless, current datasets cover a limited number of languages. With the aim to foster research in massive multilingual ST and ST for low resource language pairs, we release CoVoST 2, a large-scale multilingual ST corpus covering translations from 21 languages into English and from English into 15 languages. This represents the largest open dataset available to date from total volume and language coverage perspective. Data sanity checks provide evidence about the quality of the data, which is released under CC0 license. We also provide extensive speech recognition, bilingual and multilingual machine translation and ST baselines with open-source implementation¹.

described so far involve European languages that are in general high resource from the perspective of machine translation (MT) and speech. CoVoST is a multilingual and diversified ST corpus from 11 languages into English, based on the Common Voice project (Ardila et al., 2020). Unlike previous corpora, it involves low resource languages such as Mongolian and it also enables many-to-one ST research. Nevertheless, for all corpora described so far, the number of languages involved is limited.

In this paper, we describe CoVoST 2, an extension of CoVoST (Wang et al., 2020a) that provides translations from English (En) into 15 languages—Arabic (Ar), Catalan (Ca), Welsh (Cy), German (De), Estonian (Et), Persian (Fa), Indonesian (Id), Japanese (Ja), Latvian (Lv), Mongolian (Mn), Slovenian (Sl), Swedish (Sv), Tamil (Ta), Turkish (Tr), Chinese (Zh)—and from 21 languages into English, including the 15 target languages as well as Spanish (Es), French (Fr), Italian (It), Dutch (Nl), Portuguese (Pt), Russian (Ru). The overall speech duration is extended from 700 hours to 2880 hours. The total number of speakers is increased from 11K to 78K. We make data available at https://github.com/facebookresearch/covost under CC0 license.

## 1 Introduction

The development of benchmark datasets, such as MuST-C (Di Gangi et al., 2019), Europarl-ST (Iranzo-Sánchez et al., 2020) or CoVoST (Wang et al., 2020a), has greatly contributed to the increasing popularity of speech-to-text translation (ST) as a research topic. MuST-C provides TED talks translations from English into 8 European languages, with data amounts ranging from 385 hours to 504 hours, thereby encouraging research into end-to-end ST (Berard et al., 2016) as well as one-to-many multilingual ST (Di Gangi et al., 2019). Europarl-ST offers translations between 6 European languages, with a total of 30 translation directions, enabling research into many-to-many multilingual ST (Inaguma et al., 2019). The two corpora

---

¹ Equal contribution.
¹ https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text

## 2 Dataset Creation

### 2.1 Data Collection and Quality Control

Translations are collected from professional translators the same way as for CoVoST. We then conduct sanity checks based on language model perplexity, LASER (Artetxe and Schwenk, 2019) scores and a length ratio heuristic in order to ensure the quality of the translations. Length ratio and LASER score checks are conducted as in the original version of CoVoST. For language model perplexity checks, 20M lines are sam-


<!-- page 1 end -->


<table>
<tr>
<th></th>
<th colspan="3">Hours (CoVoST ext.)</th>
<th colspan="3">Speakers (CoVoST ext.)</th>
<th colspan="3">Src./Tgt. Tokens</th>
</tr>
<tr>
<td></td>
<td>Train</td>
<td>Dev</td>
<td>Test</td>
<td>Train</td>
<td>Dev</td>
<td>Test</td>
<td>Train</td>
<td>Dev</td>
<td>Test</td>
</tr>
<tr>
<td colspan="10">X→En</td>
</tr>
<tr>
<td>Fr</td>
<td>180(264)</td>
<td>22(23)</td>
<td>23(24)</td>
<td>2K(2K)</td>
<td>2K(2K)</td>
<td>4K(4K)</td>
<td>2M/2M</td>
<td>0.1M/0.1M</td>
<td>0.1M/0.1M</td>
</tr>
<tr>
<td>De</td>
<td>119(184)</td>
<td>21(23)</td>
<td>22(120)</td>
<td>1K(1K)</td>
<td>1K(1K)</td>
<td>4K(5K)</td>
<td>1M/1M</td>
<td>0.1M/0.2M</td>
<td>0.8M/0.8M</td>
</tr>
<tr>
<td>Es</td>
<td>97(113)</td>
<td>22(22)</td>
<td>23(23)</td>
<td>1K(1K)</td>
<td>2K(2K)</td>
<td>4K(4K)</td>
<td>0.7M/0.8M</td>
<td>0.1M/0.1M</td>
<td>0.1M/0.1M</td>
</tr>
<tr>
<td>Ca</td>
<td>81(136)</td>
<td>19(21)</td>
<td>20(25)</td>
<td>557(557)</td>
<td>722(722)</td>
<td>2K(2K)</td>
<td>0.9M/1M</td>
<td>0.1M/0.1M</td>
<td>0.2M/0.2M</td>
</tr>
<tr>
<td>It</td>
<td>28(44)</td>
<td>14(15)</td>
<td>15(15)</td>
<td>236(236)</td>
<td>640(640)</td>
<td>2K(2K)</td>
<td>0.3M/0.3M</td>
<td>89K/95K</td>
<td>88K/93K</td>
</tr>
<tr>
<td>Ru</td>
<td>16(18)</td>
<td>10(15)</td>
<td>11(14)</td>
<td>8(8)</td>
<td>30(30)</td>
<td>417(417)</td>
<td>0.1M/0.1M</td>
<td>89K/0.1M</td>
<td>81K/0.1M</td>
</tr>
<tr>
<td>Zh</td>
<td>10(10)</td>
<td>8(8)</td>
<td>8(8)</td>
<td>22(22)</td>
<td>83(83)</td>
<td>784(784)</td>
<td>0.1M/85K</td>
<td>91K/60K</td>
<td>88K/57K</td>
</tr>
<tr>
<td>Pt</td>
<td>7(10)</td>
<td>4(5)</td>
<td>5(6)</td>
<td>2(2)</td>
<td>16(16)</td>
<td>301(301)</td>
<td>67K/68K</td>
<td>27K/28K</td>
<td>34K/34K</td>
</tr>
<tr>
<td>Fa</td>
<td>5(49)</td>
<td>5(11)</td>
<td>5(40)</td>
<td>532(545)</td>
<td>854(908)</td>
<td>1K(1K)</td>
<td>0.3M/0.3M</td>
<td>67K/73K</td>
<td>0.2M/0.3M</td>
</tr>
<tr>
<td>Et</td>
<td>3(3)</td>
<td>3(3)</td>
<td>3(3)</td>
<td>20(20)</td>
<td>74(74)</td>
<td>135(135)</td>
<td>23K/32K</td>
<td>19K/27K</td>
<td>20K/27K</td>
</tr>
<tr>
<td>Mn</td>
<td>3(3)</td>
<td>3(3)</td>
<td>3(3)</td>
<td>4(4)</td>
<td>24(24)</td>
<td>209(209)</td>
<td>20K/23K</td>
<td>19K/22K</td>
<td>18K/20K</td>
</tr>
<tr>
<td>Nl</td>
<td>2(7)</td>
<td>2(3)</td>
<td>2(3)</td>
<td>74(74)</td>
<td>144(144)</td>
<td>379(383)</td>
<td>58K/59K</td>
<td>19K/19K</td>
<td>20K/20K</td>
</tr>
<tr>
<td>Tr</td>
<td>2(4)</td>
<td>2(2)</td>
<td>2(2)</td>
<td>34(34)</td>
<td>76(76)</td>
<td>324(324)</td>
<td>24K/33K</td>
<td>11K/16K</td>
<td>11K/15K</td>
</tr>
<tr>
<td>Ar</td>
<td>2(2)</td>
<td>2(2)</td>
<td>2(2)</td>
<td>6(6)</td>
<td>13(13)</td>
<td>113(113)</td>
<td>10K/13K</td>
<td>9K/11K</td>
<td>8K/10K</td>
</tr>
<tr>
<td>Sv</td>
<td>2(2)</td>
<td>1(1)</td>
<td>2(2)</td>
<td>4(4)</td>
<td>7(7)</td>
<td>83(83)</td>
<td>12K/12K</td>
<td>8K/9K</td>
<td>9K/10K</td>
</tr>
<tr>
<td>Lv</td>
<td>2(2)</td>
<td>1(1)</td>
<td>2(2)</td>
<td>2(2)</td>
<td>3(3)</td>
<td>54(54)</td>
<td>11K/14K</td>
<td>6K/7K</td>
<td>8K/10K</td>
</tr>
<tr>
<td>Sl</td>
<td>2(2)</td>
<td>1(1)</td>
<td>1(1)</td>
<td>2(2)</td>
<td>1(1)</td>
<td>28(28)</td>
<td>11K/13K</td>
<td>3K/4K</td>
<td>2K/2K</td>
</tr>
<tr>
<td>Ta</td>
<td>2(2)</td>
<td>1(1)</td>
<td>1(1)</td>
<td>3(3)</td>
<td>2(2)</td>
<td>48(48)</td>
<td>6K/10K</td>
<td>2K/3K</td>
<td>3K/5K</td>
</tr>
<tr>
<td>Ja</td>
<td>1(1)</td>
<td>1(1)</td>
<td>1(1)</td


<!-- page 2 end -->


<table>
<tr>
<th></th>
<th>ASR</th>
<th>MT</th>
<th>+Rev<sup>†</sup></th>
<th>C-ST</th>
<th>+Rev<sup>†</sup></th>
<th>E-ST</th>
<th>ST</th>
<th>MT</th>
<th>+Rev<sup>†</sup></th>
<th>C-ST</th>
<th>+Rev<sup>†</sup></th>
<th>E-ST</th>
<th>ST</th>
</tr>
<tr>
<td>En</td>
<td>25.6</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Fr</td>
<td>18.3</td>
<td>37.9</td>
<td>38.1</td>
<td>27.6</td>
<td>27.6</td>
<td>24.3</td>
<td>26.3</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>De</td>
<td>21.4</td>
<td>28.2</td>
<td>31.2</td>
<td>21.0</td>
<td>22.6</td>
<td>8.4</td>
<td>17.1</td>
<td>29.0</td>
<td>29.1</td>
<td>18.3</td>
<td>18.1</td>
<td>13.6</td>
<td>16.3</td>
</tr>
<tr>
<td>Es</td>
<td>16.0</td>
<td>36.3</td>
<td>36.2</td>
<td>27.4</td>
<td>27.4</td>
<td>12.0</td>
<td>23.0</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Ca</td>
<td>12.6</td>
<td>24.9</td>
<td>31.1</td>
<td>21.3</td>
<td>25.1</td>
<td>14.4</td>
<td>18.8</td>
<td>38.8</td>
<td>38.6</td>
<td>24.1</td>
<td>24.1</td>
<td>20.2</td>
<td>21.8</td>
</tr>
<tr>
<td>It</td>
<td>27.4</td>
<td>19.2</td>
<td>19.0</td>
<td>13.5</td>
<td>13.5</td>
<td>0.2</td>
<td>11.3</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Ru</td>
<td>31.4</td>
<td>19.8</td>
<td>19.4</td>
<td>16.8</td>
<td>16.8</td>
<td>1.2</td>
<td>14.8</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Zh<sup>*</sup></td>
<td>45.0</td>
<td>7.6</td>
<td>16.6</td>
<td>7.0</td>
<td>9.9</td>
<td>1.4</td>
<td>5.8</td>
<td>35.3</td>
<td>38.9</td>
<td>24.6</td>
<td>25.9</td>
<td>20.6</td>
<td>25.4</td>
</tr>
<tr>
<td>Pt</td>
<td>44.6</td>
<td>14.6</td>
<td>13.9</td>
<td>9.2</td>
<td>9.2</td>
<td>0.5</td>
<td>6.1</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Fa</td>
<td>62.4</td>
<td>2.4</td>
<td>15.1</td>
<td>2.1</td>
<td>7.2</td>
<td>1.9</td>
<td>3.7</td>
<td>20.1</td>
<td>20.0</td>
<td>13.8</td>
<td>13.8</td>
<td>11.5</td>
<td>13.1</td>
</tr>
<tr>
<td>Et</td>
<td>65.7</td>
<td>0.3</td>
<td>13.7</td>
<td>0.2</td>
<td>4.4</td>
<td>0.1</td>
<td>0.1</td>
<td>24.0</td>
<td>24.3</td>
<td>14.5</td>
<td>14.5</td>
<td>11.1</td>
<td>13.2</td>
</tr>
<tr>
<td>Mn</td>
<td>65.2</td>
<td>0.2</td>
<td>5.4</td>
<td>0.1</td>
<td>1.9</td>
<td>0.1</td>
<td>0.2</td>
<td>16.8</td>
<td>17.1</td>
<td>11.0</td>
<td>10.7</td>
<td>6.6</td>
<td>9.2</td>
</tr>
<tr>
<td>Nl</td>
<td>52.8</td>
<td>2.6</td>
<td>2.5</td>
<td>1.8</td>
<td>1.8</td>
<td>0.3</td>
<td>3.0</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Tr</td>
<td>51.2</td>
<td>1.1</td>
<td>25.9</td>
<td>0.8</td>
<td>12.0</td>
<td>0.7</td>
<td>3.6</td>
<td>20.0</td>
<td>19.7</td>
<td>11.8</td>
<td>11.5</td>
<td>8.9</td>
<td>10.0</td>
</tr>
<tr>
<td>Ar</td>
<td>63.3</td>
<td>0.1</td>
<td>34.7</td>
<td>0.1</td>
<td>12.3</td>
<td>0.3</td>
<td>4.3</td>
<td>21.6</td>
<td>21.6</td>
<td>14.0</td>
<td>13.9</td>
<td>8.7</td>
<td>12.1</td>
</tr>
<tr>
<td>Sv</td>
<td>65.5</td>
<td>0.2</td>
<td>37.7</td>
<td>0.1</td>
<td>8.4</td>
<td>0.2</td>
<td>2.7</td>
<td>39.4</td>
<td>39.2</td>
<td>24.6</td>
<td>24.4</td>
<td>20.1</td>
<td>21.8</td>
</tr>
<tr>
<td>Lv</td>
<td>51.8</td>
<td>0.2</td>
<td>19.6</td>
<td>0.2</td>
<td>9.1</td>
<td>0.1</td>
<td>2.5</td>
<td>22.5</td>
<td>22.9</td>
<td>14.4</td>
<td>14.4</td>
<td>11.5</td>
<td>13.0</td>
</tr>
<tr>
<td>Sl</td>
<td>59.1</td>
<td>0.1</td>
<td>29.2</td>
<td>0.0</td>
<td>10


<!-- page 3 end -->


| Fr | De | Es | Ca | Nl | Tr | Ar | Sv | Lv | Sl | Ta | Ja | Id | Cy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bi ST | 26.3 | 17.1 | 23.0 | 18.8 | 3.0 | 3.6 | 4.3 | 2.7 | 2.5 | 3.0 | 0.3 | 1.5 | 2.5 | 2.7 |
| ASR-M† | 20.1 | 21.3 | 15.4 | 13.1 | 41.9 | 46.8 | 59.7 | 56.0 | 51.7 | 89.6 | 88.7 | 58.2 | 57.5 |
| ASR-L‡ | 19.0 | 20.2 | 14.4 | 12.5 | 46.5 | 45.6 | 54.9 | 54.8 | 44.9 | 45.2 | 78.5 | 59.4 | 48.2 | 58.7 |
| A2E MT¹ | 38.0 | 27.0 | 38.2 | 29.8 | 13.5 | 9.2 | 17.3 | 22.0 | 10.2 | 9.3 | 1.1 | 6.3 | 18.7 | 10.0 |
| A2A MT² | 40.9 | 31.7 | 41.0 | 32.4 | 19.0 | 12.1 | 17.9 | 27.0 | 11.8 | 9.5 | 1.0 | 6.5 | 23.5 | 14.1 |
| † + 1 | 27.3 | 20.0 | 28.8 | 24.9 | 8.5 | 7.1 | 10.1 | 8.8 | 6.5 | 4.9 | 0.2 | 2.8 | 7.6 | 4.9 |
| † + 1 | 28.0 | 20.6 | 29.4 | 25.2 | 8.2 | 7.6 | 10.9 | 10.3 | 6.4 | 6.4 | 0.3 | 3.5 | 9.6 | 4.9 |
| † + 2 | 28.4 | 22.7 | 30.7 | 26.6 | 11.3 | 8.7 | 10.8 | 10.3 | 6.4 | 5.3 | 0.3 | 2.8 | 9.4 | 7.8 |
| † + 2 | 29.1 | 23.2 | 31.1 | 27.2 | 10.4 | 9.3 | 12.3 | 11.9 | 7.2 | 7.0 | 0.4 | 3.8 | 11.8 | 7.4 |
| A2E-M | 27.0 | 18.9 | 28.0 | 23.9 | 6.3 | 2.4 | 0.6 | 0.8 | 0.6 | 0.6 | 0.1 | 0.2 | 0.3 | 2.5 |
| A2E-L | 26.9 | 17.6 | 26.3 | 22.1 | 4.5 | 2.7 | 0.6 | 0.6 | 0.4 | 1.2 | 0.1 | 0.2 | 0.3 | 2.6 |
| A2A-M | 22.6 | 15.6 | 23.7 | 21.1 | 8.4 | 2.8 | 0.6 | 1.2 | 0.7 | 1.1 | 0.1 | 0.2 | 0.4 | 2.5 |
| A2A-L | 26.0 | 18.9 | 27.0 | 24.0 | 8.4 | 3.7 | 0.7 | 1.2 | 0.8 | 0.6 | 0.1 | 0.3 | 0.2 | 3.3 |

Table 3: Test WER for multilingual ASR and test BLEU for multilingual X→En MT/ST. Fr, De, Es and Ca are high-resource and the rest (the right section) are low-resource. For ASR/ST, we apply temperature-based (T=2) sampling (Arivazhagan et al., 2019) to improve low-resource directions. †‡ Multilingual models trained on all 22 languages. They are also used to pre-trained ST encoders.

| Fr | De | Ca | Zh | Fa | Et | Mn | Tr | Ar | Sv | Lv | Sl | Ta | Ja | Id | Cy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bi. ST | 16.5 | 22.1 | 25.7 | 13.5 | 13.4 | 9.2 | 10.2 | 12.4 | 22.3 | 13.1 | 16.1 | 11.2 | 29.6 | 20.8 | 24.1 |
| ASR-M† | | | | | | | | | | | | | | | | |
| ASR-L‡ | | | | | | | | | | | | | | | | |
| E2A MT¹ | | | | | | | | | | | | | | | | |
| A2A MT² | | | | | | | | | | | | | | | | |
| † + 1 | | | | | | | | | | | | | | | | |
| † + 1 | | | | | | | | | | | | | | | | |
| † + 2 | | | | | | | | | | | | | | | | |
| † + 2 | | | | | | | | | | | | | | | | |
| E2A-M | | | | | | | | | | | | | | | | |
| E2A-L | | | | | | | | | | | | | | | | |
| A2A-M | | | | | | | | | | | | | | | | |
| A2A-L | | | | | | | | | | | | | | | | |

Table 4: Test WER for multilingual ASR and test BLEU for multilingual En→X MT/ST (all directions have equal resource). †‡ Multilingual models trained on all 22 languages. They are also used to pre-trained ST encoders.

4.1 Experimental Settings

For all texts, we normalize the punctuation and build vocabularies with SentencePiece (Kudo and Richardson, 2018) without pre-tokenization. For ASR and ST, character vocabularies with 100% coverage are used. For bilingual MT models, BPE (Sennrich et al., 2016) vocabularies of size 5k are learned jointly on both transcripts and translations. For multilingual MT models, BPE vocabularies of size 40k are created jointly on all available source and target text. For MT and language pair s-t, we also contrast using only s-t training data and both s-t and t-s training data (we also remove any overlap between training data from t-s and development or test set from s-t; this is also done for the A2A multilingual MT setting). The latter setting is referred to as +Rev subsequently.

We extract 80-dimensional log mel-scale filter bank features (windows with 25ms size and 10ms shift) using Kaldi (Povey et al., 2011), with per-utterance CMVN (cepstral mean and variance normalization) applied. We remove training samples having more than 3,000 frames or more than 512 characters for GPU memory efficiency.

For ASR and ST, we set d_model = 256 for bilingual models and set d_model = 512 or


<!-- page 4 end -->


well as bilingual ST models with English ASR encoder, and pre-train multilingual ST models with multilingual ASR encoder. For MT, we set $l_e = l_d = 3$ for bilingual models and $l_e = l_d = 6$ for multilingual models.

We use a beam size of 5 for all models and length penalty 1. We use the best checkpoint by validation loss for MT, and average the last 5 checkpoints for ASR and ST. For MT and ST, we report case-sensitive detokenized BLEU (Papineni et al., 2002) using sacre- BLEU (Post, 2018) with default options, except for English-Chinese and English-Japanese where we report character-level BLEU. For ASR, we report character error rate (CER) on Japanese and Chinese (no word segmentation) and word error rate (WER) on the other languages using VizSeq (Wang et al., 2019). Before calculating WER (CER), sentences are tokenized by sacre- BLEU tokenizers, lowercased and with punctuation removed (except for apostrophes and hyphens).

### 4.2 Monolingual and Bilingual Baselines

Table 2 reports monolingual baselines for ASR and bilingual MT, cascaded ST (C-ST), end-to-end ST trained from scratch (E-ST) and end-to-end ST pre-trained on ASR. As expected, the quality of transcriptions and translations is very dependent on the amount of training data per language pair. The poor results obtained on low resource pairs can be improved by leveraging training data from the opposite direction for MT and C-ST. These results serve as baseline for the research community to improve upon, including methods such as multilingual training, self-supervised pre-training and semi-supervised learning.

### 4.3 Multilingual Baselines

A2E, E2E and A2A baselines are reported in Table 3 for language pairs into English and in Table 4 for language pairs out of English. Multilingual modeling is shown to be a promising direction for improving low-resource ST.

### 5 Conclusion

We introduced CoVoST 2, the largest speech-to-text translation corpus to date for language coverage and total volume, with 21 languages into English and English into 15 languages. We also provided extensive monolingual, bilingual and multilingual baselines for ASR, MT and ST. CoVoST 2 is free to use under CC0 license and enables the research community to develop methods including, but not limited to, massive multilingual modeling, ST modeling for low resource languages, self-supervision for multilingual ST, semi-supervised modeling for multilingual ST.

### References

Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. 2020. Common voice: A massively-multilingual speech corpus. In *Proceedings of The 12th Language Resources and Evaluation Conference*, pages 4218–4222, Marseille, France. European Language Resources Association.

Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen, Yuan Cao, George Foster, Colin Cherry, Wolfgang Macherey, Zhifeng Chen, and Yonghui Wu. 2019. Massively multilingual neural machine translation in the wild: Findings and challenges.

Mikel Artetxe and Holger Schwenk. 2019. Margin-based parallel corpus mining with multilingual sentence embeddings. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 3197–3203, Florence, Italy. Association for Computational Linguistics.

Alexandre Berard, Olivier Pietquin, Christophe Servan, and Laurent Besacier. 2016. Listen and translate: A proof of concept for end-to-end speech-to-text translation. In *Proceedings of the 2016 NeurIPS Workshop on End-to-end Learning for Speech and Audio Processing*.

M. A. Di Gangi, M. Negri, and M. Turchi. 2019. One-to-many multilingual end-to-end speech translation. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 585–592.

Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2019. MuST-C: a Multilingual Speech Translation Corpus. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2012–2017, Minneapolis, Minnesota. Association for Computational Linguistics.

H. Inaguma, K. Duh, T. Kawahara, and S. Watanabe. 2019. Multilingual end-to-end speech translation. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 570–577.

J. Iranzo-Sánchez, J. A. Silvestre-Cerdà, J. Jorge, N. Roselló, A. Giménez, A. Sanchis, J. Civera, and


<!-- page 5 end -->


A. Juan. 2020. Europarl-st: A multilingual corpus for speech translation of parliamentary debates. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8229–8233.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.

Nathan Ng, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli, and Sergey Edunov. 2019. Facebook FAIR’s WMT19 news translation task submission. In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 314–319, Florence, Italy. Association for Computational Linguistics.

Pedro Javier Ortiz Suárez, Laurent Romary, and Benoît Sagot. 2020. A monolingual approach to contextualized word embeddings for mid-resource languages. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1703–1714, Online. Association for Computational Linguistics.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of NAACL-HLT 2019: Demonstrations.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics, pages 311–318. Association for Computational Linguistics.

Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and Quoc V Le. 2019. Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186–191, Brussels, Belgium. Association for Computational Linguistics.

Daniel Povey, Arnab Ghoshal, Gilles Boulianne, Lukas Burget, Ondrej Glembek, Nagendra Goel, Mirko Hannemann, Petr Motlicek, Yanmin Qian, Petr Schwarz, et al. 2011. The kaldi speech recognition toolkit. In IEEE 2011 workshop on automatic speech recognition and understanding. IEEE Signal Processing Society.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany. Association for Computational Linguistics.

Gabriel Synnaeve, Qiantong Xu, Jacob Kahn, Tatiana Likhomanenko, Edouard Grave, Vineel Pratap, Anuroop Sriram, Vitaliy Liptchinsky, and Ronan Collobert. 2020. End-to-end asr: from supervised to semi-supervised learning with mo...

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008.

Changhan Wang, Anirudh Jain, Danlu Chen, and Jiatao Gu. 2019. Vizseq: A visual analysis toolkit for text generation tasks. EMNLP-IJCNLP 2019, page 253.

Changhan Wang, Juan Pino, Anne Wu, and Jiatao Gu. 2020a. CoVoST: A diverse multilingual speech-to-text translation corpus. In Proceedings of The 12th Language Resources and Evaluation Conference, pages 4197–4203, Marseille, France. European Language Resources Association.

Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, and Juan Pino. 2020b. fairseq s2t: Fast speech-to-text modeling with fairseq. arXiv preprint arXiv:2010.05171.
