/* dev fixture: presence dock / status pill over the legacy `clock` display */
BB.fixture = {
 "v": 1,
 "display": "clock",
 "now": "2026-11-24T16:02:00-08:00",
 "people": [
  "Jacob",
  "Carina",
  "Erik",
  "Mia"
 ],
 "atmosphere": {
  "phase": "golden",
  "weather": "partly",
  "time": "16:02"
 },
 "layout": {
  "type": "column",
  "align": "center",
  "padding": [
   0,
   0,
   0,
   0
  ],
  "children": [
   {
    "type": "clock",
    "size": "xl"
   }
  ]
 },
 "data": {},
 "args": {},
 "status": null,
 "presence": {
  "state": "listening",
  "who": "Erik",
  "text": "Can you remind me to bring my swim bag and the permission slip and also tell Mom that practice ends at seven tonight"
 }
};
