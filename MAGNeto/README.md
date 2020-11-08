# MAGNeto

*Authors: Phung Trong Hieu, Vu Tuan Anh, Nguyen Dinh Tung, Do Thanh Lam, Tran Thanh Trung, Ngo Nam Giang (LabTeam - PixtaVietnam)*

---

## Downloading dataset
- Official: https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
- Unofficial: http://cs-people.bu.edu/hekun/data/TALR/NUSWIDE.zip

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
