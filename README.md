[Dataset](https://drive.google.com/file/d/1oIMuH1rc87GxBR86FQocNHSmZW1i8DXH/view?usp=sharing)

The script for getting dataset

```bash
mkdir data
mkdir models
mkdir result
cd data
filename='data.zip'
fileid='1oIMuH1rc87GxBR86FQocNHSmZW1i8DXH'
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O ${filename}


# Unzip
unzip -q ${filename}
rm ${filename}
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
