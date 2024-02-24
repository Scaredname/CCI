[dataset](https://drive.google.com/file/d/1oIMuH1rc87GxBR86FQocNHSmZW1i8DXH/view?usp=sharing)

To get dataset

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
cd
```
