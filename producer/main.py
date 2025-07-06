from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from kafka import KafkaProducer
import json, time, signal, os, threading
from datetime import datetime

KAFKA_TOPIC = "spy_option_ticks"
KAFKA_BROKER = os.getenv("BOOTSTRAP_SERVERS", "kafka:9092")
SPOT_REQ_ID = 999
STRIKE_WINDOW = 10
PRINT_INTERVAL = 10

def create_kafka_producer(bootstrap_servers, retries=10, delay=5):
    for attempt in range(retries):
        try:
            print(f"📱 Connecting to Kafka... Attempt {attempt + 1}")
            p = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            print("✅ Kafka connected.")
            return p
        except Exception as e:
            print(f"⏳ Kafka not ready: {e}")
            time.sleep(delay)
    print("❌ Failed to connect to Kafka.")
    exit(1)

producer = create_kafka_producer(KAFKA_BROKER)

class IBKafkaProducer(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.running = True
        self.spot_price = None
        self.active_subs = set()
        self.calls_data = {}
        self.puts_data = {}
        self.last_print_time = 0
        self.ticker_id_counter = 10000
        self.ticker_id_map = {}
        self.reverse_ticker_id = {}
        self._refresh_thread_started = False
        self.refresh_interval = 10  # seconds

        self.maturities = [
            "20250701"
        ]

    def _get_ticker_id(self, maturity, strike, right):
        key = (maturity, strike, right)
        if key not in self.ticker_id_map:
            self.ticker_id_map[key] = self.ticker_id_counter
            self.reverse_ticker_id[self.ticker_id_counter] = key
            self.ticker_id_counter += 1
        return self.ticker_id_map[key]

    def nextValidId(self, orderId):
        print(f"✅ nextValidId received: {orderId}")
        self.reqMarketDataType(1)#*****************************************************
        self.reqMktData(SPOT_REQ_ID, self._make_spot_contract(), "", False, False, [])

    def _make_spot_contract(self):
        c = Contract()
        c.symbol = "SPY"
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    def _make_opt_contract(self, maturity, strike, right):
        c = Contract()
        c.symbol = "SPY"
        c.secType = "OPT"
        c.exchange = "SMART"
        c.currency = "USD"
        c.lastTradeDateOrContractMonth = maturity
        c.strike = strike
        c.right = right
        c.multiplier = "100"
        c.tradingClass = "SPY"
        return c

    def tickPrice(self, reqId, tickType, price, attrib):
        if reqId == SPOT_REQ_ID and tickType == 4:
            self.spot_price = price
            if not self._refresh_thread_started:
                self._refresh_thread_started = True
                threading.Thread(target=self._refresh_loop, daemon=True).start()
            return

        key = self.reverse_ticker_id.get(reqId)
        if not key:
            return
        maturity, strike, right = key
        store = self.calls_data if right == "C" else self.puts_data
        store.setdefault((maturity, strike), {})
        if tickType == 1: store[(maturity, strike)]['bid'] = price
        if tickType == 2: store[(maturity, strike)]['ask'] = price
        self._maybe_send()

    def tickOptionComputation(self, reqId, tickType, tickAttrib,
                                impliedVol, delta, optPrice,
                                gamma, vega, theta, undPrice, pvDividend):
        if tickType != 13:
            return
        key = self.reverse_ticker_id.get(reqId)
        if not key:
            return
        maturity, strike, right = key
        store = self.calls_data if right == "C" else self.puts_data
        store.setdefault((maturity, strike), {})
        store[(maturity, strike)].update({
            'iv': impliedVol,
            'price': optPrice,
            'delta': delta,
            'vega': vega,
            'theta': theta
        })
        self._maybe_send()

    def _refresh_loop(self):
        while self.running:
            time.sleep(self.refresh_interval)
            self._update_option_requests()

    def _update_option_requests(self):
        if self.spot_price is None:
            return
        center = round(self.spot_price)
        print(f"🔄 Refreshing strikes around {center}")

        for m, st in self.active_subs:
            self.cancelMktData(self._get_ticker_id(m, st, "C"))
            self.cancelMktData(self._get_ticker_id(m, st, "P"))
        self.active_subs.clear()

        self.ticker_id_map.clear()
        self.reverse_ticker_id.clear()
        self.ticker_id_counter = 10000

        for m in self.maturities:
            for offset in range(-STRIKE_WINDOW, STRIKE_WINDOW + 1):
                strike = center + offset
                cid = self._get_ticker_id(m, strike, "C")
                pid = self._get_ticker_id(m, strike, "P")
                self.reqMktData(cid, self._make_opt_contract(m, strike, "C"), "", False, False, [])
                self.reqMktData(pid, self._make_opt_contract(m, strike, "P"), "", False, False, [])
                self.active_subs.add((m, strike))

    def _maybe_send(self):
        now = time.time()
        if now - self.last_print_time < PRINT_INTERVAL:
            return
        self.last_print_time = now
        current_time = datetime.utcnow().isoformat()

        for m, st in sorted(self.active_subs):
            call = self.calls_data.get((m, st), {})
            put = self.puts_data.get((m, st), {})
            if all(k in call for k in ('bid', 'ask', 'iv')) and all(k in put for k in ('bid', 'ask', 'iv')):
                msg = {
                    "time": current_time,
                    "spot_price": self.spot_price,
                    "maturity": m,
                    "strike": st,
                    "call_bid": call['bid'],
                    "call_ask": call['ask'],
                    "call_iv": call['iv'],
                    "call_delta": call.get('delta'),
                    "call_vega": call.get('vega'),
                    "call_theta": call.get('theta'),
                    "put_bid": put['bid'],
                    "put_ask": put['ask'],
                    "put_iv": put['iv'],
                    "put_delta": put.get('delta'),
                    "put_vega": put.get('vega'),
                    "put_theta": put.get('theta')
                }
                try:
                    producer.send(KAFKA_TOPIC, msg)
                    print("📤 Sent:", msg)
                except Exception as e:
                    print(f"⚠️ Failed to send: {e}")

    def error(self, reqId, errorCode, errorString):
        if errorCode not in (2104, 2106, 2158):
            print(f"❌ ERROR {reqId} {errorCode}: {errorString}")

def shutdown(signum, frame):
    print("🚩 Shutting down...")
    app.running = False
    app.disconnect()
    producer.flush()
    exit(0)

if __name__ == "__main__":
    app = IBKafkaProducer()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    app.connect("192.168.0.86", 7497, clientId=10)
    print("✅ Connected to TWS. Waiting for data...")

    threading.Thread(target=app.run, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)
