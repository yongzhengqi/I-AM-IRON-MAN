# I AM IRON MAN

This is an advanced facial filter to let you become Iron Man which implemented 3D reconstruction and rendering to handle large-pose cases.

![https://ml.qizy.tech/wp-content/uploads/2019/05/2363721637431-1200x800.png](https://ml.qizy.tech/wp-content/uploads/2019/05/2363721637431-1200x800.png)

## Getting Started

1. Clone this repo to your local machine
2. Install required packages

```shell
pip install tqdm numpy trimesh pyrender
```

3. Run this project

```shell
python3 i_am_iron_man.py
```

4. If any required retrained model is missing, use the following command to sepicify its location

``` shell
python3 i_am_iron_man.py -e lib/models/Mark42.obj -m lib/models/mobilenet.model -d lib/models/68_face_landmarks.dat
```



## Authors

* Charles Young: [Qizy's Sites](https://qizy.tech/)
* Zhenbang You
* Chengzhi Yi

## License

This project is licensed under the MIT License.

## Acknowledgments

* Thanks for Jianzhu Guo's fantastic work [3DDFA](https://github.com/cleardusk/3DDFA).
* Thanks for mmatl's amazing render [pyrender](https://github.com/mmatl/pyrender).


