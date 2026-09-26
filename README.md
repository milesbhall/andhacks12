[Project Name] is a low-latency pipeline that transcribes speeches and social posts from market-moving public figures 
(e.g. Fed Chair Kevin Warsh, Donald Trump) 
in real time, then scores each statement for "surprise" against that speaker's own historical baseline rather than generic sentiment — 
flagging when someone deviates meaningfully from what they've said before. 
When a high-surprise statement is detected, the system maps it to the relevant Kalshi contract and can execute a trade automatically, 
logging every prediction against its actual outcome to build a calibrated track record over time.

Why it exists

Tools like Bloomberg Terminal surface news and generic sentiment to a human trader, 
but don't execute trades and don't measure deviation from a specific speaker's own history — 
they tell you "this is negative news," not "this is unusually hawkish for this person specifically." 
Academic research (Chicago Fed, FedSpeak Decoder, and others) has shown that speaker-relative surprise — not raw sentiment — 
is what actually predicts market reaction, but that work has stayed offline and backward-looking. 
Nothing connects that finding to a live feed or a tradeable market. 
This project closes that gap: it's the first system to run speaker-baseline surprise scoring live, 
end-to-end, against an actual prediction market.
