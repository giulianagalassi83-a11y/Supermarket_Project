# Matching audit — how to label

**Files**
- `audit_felipe.xlsx`: 100 pairs.
- `audit_teammate.xlsx`: 40 of the same pairs, for inter-rater κ.
- `audit_key.csv`: the hidden answers (engine decision, Gemini's label). **Don't open it until you've finished.**

**Work alone.** Don't compare answers with each other, and don't look at Gemini's labels.

## Steps
1. Open your spreadsheet. Each row is one pair: product A and product B, with chain, name, brand, size and a link to the product page.
2. In the `label` column, pick one value from the dropdown. Use the definitions below.
3. If something is ambiguous, write a short reason in `notes` (e.g. "size not shown", "link dead").
4. Save the file under the same name and in the same place.
5. Run `python -m src.similarity.audit score` → `data/similarity/results/kappa.csv` and `audit_matches.csv`.

About 20 minutes for 100 pairs. Open a link only when the name and size don't settle it.

## Labels (the same definitions Gemini was given)
| label | when |
|---|---|
| `same` | Same product: same brand (or both the own brand of their chain), same variant, size within ±2 %, same pack count. |
| `equivalent` | A fair substitute: same type of product, no variant conflict (whole vs skimmed milk, extra virgin vs refined oil, sugar vs zero), per-item size at most 2× apart (1 L vs 1.5 L is fine; a multipack of the same item counts). The brand may differ. |
| `different` | Anything else: another product type, a variant conflict, or a size more than 2× apart. |
| `unsure` | You can't tell, even from the link. Use it sparingly; these rows are left out of κ. |

**Tips**
- An empty size means the shop doesn't state it. Judge from the name and the link.
- Brand spelling doesn't matter ("G. Blanca" = "Gallina Blanca"). A sub-brand of the same maker counts as the same brand (Activia / Danone).
