# Trading Bot

## Getting Started

Use this directory as your main directory.

Run `backtest.sh` to backtest the strategy.

Before your first backtest, make sure you have the data downloaded via git-lfs, using the `git-lfs pull` command.

In the `backtest.sh` script, the lines look like `freqtrade backtesting ...`. I intentionally use plain freqtrade commands instead of their recommended docker-compose setup as its faster without the docker overhead. But when getting started, just use their docker-compose setup. To do, whereever it says `freqtrade ...`, replace it with `docker compose run --rm freqtrade ...`. This will pull the docker image
