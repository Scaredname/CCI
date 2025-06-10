The script before starting experiments

```bash
mkdir models
mkdir result
filename='data.zip'

# Unzip
unzip -q ${filename}
```

You can run the experiments by the following bash script.

```bash
COUNT=1
MODULE=test_initializer.py
DATASET=a

while [ $COUNT -gt 0 ]; do
    echo "#### Left $COUNT ####"
    # python "$MODULE" -d "$DATASET" -m distmult -lr 0.001 -ba -de test_multiple
    python "$MODULE" -d "$DATASET" -m RotatE -lr 0.001 -wcg 50
    # python "$MODULE" -d "$DATASET" -m complex -lr 0.001 -cg 50 -np
    # python "$MODULE" -d "$DATASET" -m TransE -lr 0.0001 -cg 50
    ((COUNT--))
done
```

You can run `python vertical_result.py` and `python horizontal_result.py` to generate structured experimental results.
