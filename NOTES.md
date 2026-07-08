# Fixes queue for Claude

Anything you want changed on the next edit goes here (or just tell me in chat — either works).
Before each edit I pull this file, do the open items, and tick them off. Keeping everything in this **one** file means nothing gets scattered across ad-hoc notes.

**Tips**
- You don't need to retype tracebacks. After a failed Colab run, **File → Save a copy in GitHub** (Colab commits the cell outputs), then add an item like "fix the error in cell N" — I'll read the actual traceback out of the notebook when I pull.
- List a warning here and I'll treat it as "please fix or suppress." Unsure whether a warning matters? Add it anyway and I'll assess.
- Reference cells by number or by a short quote of their first line so I can find them fast.

## Open
<!-- add items below; example format: -->

## Done
<!-- I move finished items here (or you can just delete them) -->
- [x] Imports cell: dropped the fragile `transformer_lens` `utils` import (it was aborting the cell before `AutoTokenizer` imported). Inlined `get_act_name` as fixed hook-name strings rather than switching to `.utilities`, so it works on any TransformerLens version.
