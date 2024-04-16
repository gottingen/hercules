# <p align="center">hercules</p>

<p align="center">
    <a href="https://hercules-docs.readthedocs.io/en/latest/"><font face="黑体" color=#0099fc size=4>Documentation</font></a>|
    <a href="https://hercules-docs.readthedocs.io/en/latest/"><font face="黑体" color=#0099fc size=4>文档</font></a>|
    <a href="CONTRIBUTORS.md"><font face="黑体" color=#0099fc size=4>Contributors</font></a>|
    <a href="NEWS.md"><font face="黑体" color=#0099fc size=4>Road Map</font></a>|
</p>
<p align="center">
<img src="docs/source/image/hercules_img.png"></img>
</p>


## the hercule work flow


![pic][1]

[docs](https://hercules-docs.readthedocs.io/en/latest/)

# try jupyter in docker

![](docs/source/image/demo_jupyter.gif)

run the following command to start a jupyter notebook server in docker
the password is `123456`

```bash
    docker run -p 8888:8888 lijippy/hs_jupyter:r0.2.7 \
    /usr/local/bin/jupyter notebook --allow-root --ip 0.0.0.0
```

[1]: docs/source/image/flow.webp
[2]: docs/source/image/hercules_img.png
[3]: https://hercules-docs.readthedocs.io/en/latest/