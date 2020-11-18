Obtain dataset from [here](https://20bn.com/datasets/something-something).  
Unpack it:
(the following command works only if all dataset parts are on you device)

```
cat 20bn-something-something-v2-?? | tar zx
```

Prepare '20bn-sth-sth' dataset
1. Install 'fromdos'

```
apt install tofrodos
```

2. [Download](https://matroska.org/downloads/mkclean.html) and install mkclean

```
wget <link from site>
tar xf <downloaded file>
cd mkclean-<version>
fromdos mkclean/configure.compiled
./mkclean/configure.compiled
make -C mkclean
``` 

3. Process videos with `mkclean`
(Since 200K etries is too many for mkclean, you should use sth like `xargs` or write a for loop)

```
for f in `ls`; do <dir to unpacked mkclean>/mkclean-<version>/release/gcc\_linux\_x64/mkclean $f <output dir>/$f; done
```

The dataset is ready to use

## TensorboardX

To add videos, install moviepy package `pip install moviepy`.  
To speed up saving a large amount of data, you can optionally do

```
pip install crc32c
```
