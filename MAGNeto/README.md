# [MAGNeto: An Efficient Deep Learning Method for the Extractive Tags Summarization Problem](https://arxiv.org/abs/2011.04349)

## Downloading NUS-WIDE dataset
- Official: https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
- Unofficial: http://cs-people.bu.edu/hekun/data/TALR/NUSWIDE.zip (recommended for downloading all images)

## Data preparation

### Moving images to a single directory

```
./data/nus_wide/notebooks/Move\ Images.ipynb
```

### Preparing tag data

```
./data/nus_wide/notebooks/Prepare\ Tag\ Data.ipynb
```

## Setting up the environment

```bash
pip install -U pip
pip install -r requirements.txt
```

## Generating label for raw data

- Step 1: Reconfigure `scripts/start_preprocess.sh`

    To list all configurable parameters, run

    ```bash
    python preprocess.py -h
    ```

- Step 2: Run

    ```bash
    bash scripts/start_preprocess.sh
    ```

## Training the model

- Step 1: Reconfigure `scripts/start_train.sh`

    To list all configurable parameters, run

    ```bash
    python train.py -h
    ```

- Step 2: Run

    ```bash
    bash scripts/start_train.sh
    ```

## Inferring test data

- Step 1: Reconfigure `scripts/start_infer.sh`

    To list all configurable parameters, run

    ```bash
    python infer.py -h
    ```

- Step 2: Run

    ```bash
    bash scripts/start_infer.sh
    ```

## Reference

Please acknowledge the following paper in case of using this code as part of any published research:

**"MAGNeto: An Efficient Deep Learning Method for the Extractive Tags Summarization Problem."**
Hieu Trong Phung, Anh Tuan Vu, Tung Dinh Nguyen, Lam Thanh Do, Giang Nam Ngo, Trung Thanh Tran, Ngoc C. Lê.

    @article{Hieu2020,
        title={MAGNeto: An Efficient Deep Learning Method for the Extractive Tags Summarization Problem},
        author={Hieu Trong Phung and Anh Tuan Vu and Tung Dinh Nguyen and Lam Thanh Do and Giang Nam Ngo and Trung Thanh Tran and Ngoc C. L\^{e}},
        journal={arXiv preprint arXiv:2011.04349},
        year={2020}
    } 

## License

The code is released under the [GPLv3 License](https://www.gnu.org/licenses/gpl-3.0.en.html).
