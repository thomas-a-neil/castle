# castle

AlphaGo-inspired agent for playing chess.

Jared Rulison, Rajiv Sambharya, Neil Thomas

UC Berkeley CS 294-112 final project Fall 2017.

# Installing requirements

```
virtualenv -p python3 venv
pip3 install -r requirements.txt
```

# Running tests

For now, add the python path to allow for relative imports. We'll try to remove this later.
```
PYTHONPATH=. py.test
```

# Installing Stockfish

castle uses the Stockfish chess engine to evaluate non-terminal chess positions.

You can download Stockfish [here](https://stockfishchess.org/download/). Unzip it and take note of the path of the executable, as you will need it later to instantiate the engine.
