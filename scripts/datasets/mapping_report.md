# Phase 0 dataset registry report

Included: **46**  |  skipped/recorded: 30

## Bucket counts (target 16 / 17 / 17)

| bucket | included | target |
|---|---|---|
| [3,7) | 16 | 16 |
| [7,11) | 17 | 17 |
| [11,15) | 13 | 17 |

## Included by source

| source | count |
|---|---|
| UCI via OpenML | 18 |
| OpenML curated suite | 13 |
| legacy cache | 8 |
| OpenML open pool | 7 |

## Median-imputed datasets (8)

- HCV_data: median-imputed 31 cells (0.5%)
- colleges_aaup: median-imputed 256 cells (1.7%)
- eucalyptus: median-imputed 448 cells (4.3%)
- marketing-income: median-imputed 2694 cells (2.5%)
- pbcseq: median-imputed 954 cells (4.1%)
- Fitness_Club: median-imputed 20 cells (0.4%)
- sick: median-imputed 2142 cells (9.5%)
- breast-w: median-imputed 16 cells (0.3%)

## UCI candidates not in the pool (22)

| candidate | UCI id | reason |
|---|---|---|
| Abalone | 1 | covered by legacy cache (ABALONE) |
| Apartment for Rent Classified | 555 | no OpenML mirror found |
| ApisTox | 995 | no OpenML mirror found |
| Assessing Mathematics Learning in Higher Education | 1031 | no OpenML mirror found |
| Auction Verification | 713 | regression-like target (2039 distinct values) |
| Banknote Authentication | 267 | covered by legacy cache (BANKNOTE AUTHENTICATION) |
| Contraceptive Method Choice | 30 | covered by legacy cache (CONTRACEPTIVE) |
| Drug Consumption (Quantified) | 373 | no OpenML mirror found |
| Electrical Grid Stability Simulated Data |  | simulated |
| Energy Efficiency | 242 | regression-like target (37 distinct values) |
| Forty Soybean Cultivars from Subsequent Harvests | 913 | no OpenML mirror found |
| Haberman's Survival | 43 | d=2 outside [3,14] |
| Iranian Churn | 563 | no OpenML mirror found |
| Land Mines | 763 | no OpenML mirror found |
| MONK's Problems |  | simulated |
| NHANES Age Prediction Subset | 887 | regression-like target (69 distinct values) |
| Raisin | 850 | no OpenML mirror found |
| Rice (Cammeo and Osmancik) | 545 | no OpenML mirror found |
| Statlog (German Credit Data) | 144 | covered by legacy cache (GERMAN CREDIT) |
| Statlog (Heart) | 145 | covered by legacy cache (HEART) |
| Thyroid Disease | 102 | covered by legacy cache (THYROID) |
| Website Phishing | 379 | d=0 outside [3,14] |

## Other skipped OpenML candidates (8)

| dataset | OpenML id | reason |
|---|---|---|
| E-CommereShippingData | 46924 | not drawn: bucket [3,7) at target |
| balance-scale | 11 | not drawn: bucket [3,7) at target |
| churn | 46915 | d=15 outside [3,14] |
| credit-approval | 29 | not drawn: bucket [3,7) at target |
| mfeat-morphological | 18 | not drawn: bucket [3,7) at target |
| phoneme | 1489 | not drawn: bucket [3,7) at target |
| vowel | 307 | not drawn: bucket [7,11) at target |
| wilt | 40983 | not drawn: bucket [3,7) at target |

## Legacy datasets outside the new pool (n < 200)

Echocardiogram (n=61), Hepatitis (n=129), Flags (n=194) stay in the legacy per-dataset appendix tables only.
