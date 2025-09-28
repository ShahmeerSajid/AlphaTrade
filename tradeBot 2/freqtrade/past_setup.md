touch setup.sh%                                                       
(base) salmanshahid@Salmans-MacBook-Pro freqtrade % touch setup.sh
(base) salmanshahid@Salmans-MacBook-Pro freqtrade % zsh setup.sh 
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:100  1230  100  1230    0     0    955      0  0:00:01  0:00:01 --:--:100  1230  100  1230    0     0    955      0  0:00:01  0:00:01 --:--:--   954
[+] Running 11/11
 ⠿ freqtrade Pulled                                                                       192.9s
   ⠿ b16f1b166780 Pull complete                                                            43.2s
   ⠿ 8a45c7e905d6 Pull complete                                                            43.3s
   ⠿ dc110bfd751c Pull complete                                                            43.9s
   ⠿ f01499923487 Pull complete                                                            44.0s
   ⠿ b437d7cef962 Pull complete                                                            44.5s
   ⠿ 4f4fb700ef54 Pull complete                                                            44.6s
   ⠿ 12626a2a5ccd Pull complete                                                            52.4s
   ⠿ cb4596deeabc Pull complete                                                           185.8s
   ⠿ 99817d26035a Pull complete                                                           186.6s
   ⠿ 8e32cb7e2e51 Pull complete                                                           186.8s
[+] Running 1/0
 ⠿ Network ft_userdata_default  Created                                                     0.1s
2025-06-27 05:55:38,233 - freqtrade - INFO - freqtrade 2025.5
2025-06-27 05:55:38,555 - numexpr.utils - INFO - NumExpr defaulting to 5 threads.
2025-06-27 05:55:40,218 - freqtrade - INFO - freqtrade 2025.5
2025-06-27 05:55:40,483 - numexpr.utils - INFO - NumExpr defaulting to 5 threads.
? Do you want to enable Dry-run (simulated trades)? Yes
? Please insert your stake currency: USDT
? Please insert your stake amount (Number or 'unlimited'): unlimited
? Please insert max_open_trades (Integer or -1 for unlimited open trades): 3
? Time Have the strategy define timeframe.
? Please insert your display Currency for reporting (leave empty to disable FIAT conversion): USD
 
? Select exchange kraken
? Do you want to enable Telegram? Yes
? Insert Telegram token
? Insert Telegram chat id
? Do you want to enable the Rest API (includes FreqUI)? Yes
? Insert Api server Listen Address (0.0.0.0 for docker, otherwise best left untouched) 0.0.0.0
? Insert api-server username freqtrader
? Insert api-server password
2025-06-27 05:58:35,623 - freqtrade.configuration.deploy_config - INFO - Writing config to 
`user_data/config.json`.
2025-06-27 05:58:35,623 - freqtrade.configuration.deploy_config - INFO - Please make sure to 
check the configuration contents and adjust settings to your needs.
(base) salmanshahid@Salmans-MacBook-Pro freqtrade % docker compose up -d
can't find a suitable configuration file in this directory or any parent: not found
(base) salmanshahid@Salmans-MacBook-Pro freqtrade % 