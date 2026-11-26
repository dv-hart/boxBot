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
  "state": "speaking",
  "who": "Jacob",
  "text": "Thursday at 6:30, at Benaroya Hall. Doors open at six, and the parking garage on Second fills early, so leave by 5:15."
 }
};
