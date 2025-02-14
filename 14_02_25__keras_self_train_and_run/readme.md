***How to run:***

1. Setup env
```
poetry env use ~/.pyenv/versions/3.9.16/bin/python
poetry shell
poetry install
```
2. Train model and save it

```
python src/model_train2.py
```
Finally in root folder will be `liveness_model.h5`

3. Run it via `poetry run src/liveness21.py`

4. Run it via `poetry run src/liveness22.py`
