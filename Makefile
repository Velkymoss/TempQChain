install:
    @if ! command -v poetry >/dev/null 2>&1; then \
        echo "Poetry not found. Installing..."; \
        pip install --user poetry; \
    fi
    poetry install

baseline-yn:
    python main.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8

primal-dual-yn:
    python main.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --pmd T --constraints T

primal-dual-qchain-yn:
    python main.py --epoch 8 --train_file SPARTUN --test_file SPARTUN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --model t5-adapter --pmd T --constraints T --save T --save_file Q_chain_T5

baseline-fr:
    python main_rel.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8

primal-dual-fr:
    python main_rel.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --pmd T --constraints T

primal-dual-qchain-fr:
    python main_rel.py --epoch 8 --train_file SPARTUN --test_file SPARTUN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --pmd T --constraints T --save T --save_file Q_chain_T5