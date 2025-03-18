# preprocess images
```
python3 scripts/preprocessing/preprocess_images.py --input-dir data/formula_images --output-dir data/processed_formula_images
```
# normalize formulas
```
python3 scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file data/im2latex_formulas.lst --output-file data/im2latex_formulas.norm.lst
```
# filter
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir data/processed_formula_images --label-path data/im2latex_formulas.norm.lst --data-path data/im2latex_train.lst --output-path data/im2latex_train_filter.lst
```
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir data/processed_formula_images --label-path data/im2latex_formulas.norm.lst --data-path data/im2latex_validate.lst --output-path data/im2latex_validate_filter.lst
```
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir data/processed_formula_images --label-path data/im2latex_formulas.norm.lst --data-path data/im2latex_test.lst --output-path data/im2latex_test_filter.lst 
```
# build vocab
```
python3 scripts/preprocessing/generate_latex_vocab.py --data-path data/im2latex_train_filter.lst --label-path data/im2latex_formulas.norm.lst --output-file data/latex_vocab.txt
```