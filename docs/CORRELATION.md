# Evidence Correlation Rules

## Purpose

`src/threat_intel/correlation.py` combines a supplied model result and provider results into an assessment while preserving each original source. It does not retrain or call the model, modify SHAP values, average probabilities, invent weights, or produce one universal malicious/benign boolean. The `ThreatAssessment` response keeps `ml`, `threat_intelligence`, and `correlation` as separate fields. Optional SHAP summary metadata remains nested within the ML evidence.

The Phase 5 model uses phishing as positive class `1` and reports its baseline class at threshold `0.5`. The engine uses the supplied model prediction and probability; it does not establish a deployment threshold.

## Deterministic rules

| Evidence | Correlation status | Interpretation |
|---|---|---|
| PhishTank record is in database, verified, and valid; ML predicts phishing | `correlated_phishing_evidence` | Strong correlated phishing evidence from distinct sources; provider result remains separately visible. |
| Same confirmed PhishTank record; ML predicts legitimate | `ml_ti_disagreement` | Explicit contradiction. Neither signal is discarded. |
| OpenPhish phishing feed listing; ML predicts phishing | `correlated_phishing_listings` | Both indicate phishing, but OpenPhish remains a feed listing and is not upgraded to a verified PhishTank confirmation. |
| PhishTank confirmed record or OpenPhish listing; ML predicts legitimate | `ml_ti_disagreement` | Explicit disagreement with evidence levels retained. |
| URLhaus malware-distribution match; ML predicts phishing | `ml_phishing_and_malware_intelligence` | Both indicate concern, but URLhaus's verdict remains malware distribution, not phishing. |
| URLhaus malware-distribution match; ML predicts legitimate | `malware_intelligence_ml_not_phishing` | Malware intelligence is surfaced despite the model result. This is not labeled a phishing disagreement because the verdicts concern different classes of threat. |
| All configured providers fail/unavailable and ML exists | `ml_only_ti_unavailable` | ML-only result with explicit TI availability warning. |
| No provider match and at least one provider unavailable | `ml_only_partial_ti` | No match among available responses; TI coverage is partial. |
| No provider matches and all queried providers respond | `ml_only_no_ti_match` | Model result is retained; absence from these sources is not a safe verdict. |
| TI evidence exists without ML | `ti_only_*` | Provider evidence is represented without inventing an ML prediction. |
| Provider record has no confirmed phishing or malware verdict | `unconfirmed_ti_record` | Record is retained but not promoted to a confirmed threat classification. |

The engine's `evidence_level` is a qualitative source status (`none`, `listed`, or `confirmed`), not a probability. A URLhaus record remains malware evidence even when its source status is `confirmed`. The URLhaus API reports `threat=malware_download`; URLhaus's policy explicitly says phishing sites/kits should not be submitted because phishing is not its malware-distribution scope. Consequently, **a URLhaus match is not a phishing verdict**.

PhishTank record membership alone is not confirmation: the provider adapter requires both the official `verified` and `valid` fields to be true before assigning the confirmed phishing verdict. OpenPhish's free community list has URL entries but no per-entry verification/confidence metadata in that feed; matches remain `listed`.

## Disagreement and missing data

Disagreement is represented with `correlation.disagreement=true` and status `ml_ti_disagreement`; the original ML prediction/probability and provider verdicts remain in the payload. A URLhaus malware record alongside a low phishing-model score is not treated as an opposing phishing verdict; it is reported as distinct malware evidence. Provider `no_match`, `unavailable`, and `error` statuses remain distinguishable. If every TI source is unavailable, the assessment says ML-only and does not imply TI cleared the URL.

The rules are deterministic and deliberately qualitative. They are not calibrated risk scores and do not claim a universal security classification. Correlation reflects evidence returned at lookup time and source coverage; it cannot validate real-world maliciousness or guarantee a URL is safe.
