import asyncio
import aiohttp
import hmac
import hashlib
import time
from collections import deque, defaultdict
from datetime import datetime
import os
import json
from statistics import stdev, mean

class BinancePumpDetector:
    def __init__(self, telegram_token, telegram_chat_id):
        self.base_url = "https://api.binance.com"
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id

        # Price history for volume analysis
        self.price_history = defaultdict(lambda: deque(maxlen=60))
        self.volume_history = defaultdict(lambda: deque(maxlen=60))
        self.trade_history = defaultdict(lambda: deque(maxlen=100))
        self.orderbook_history = defaultdict(lambda: deque(maxlen=30))

        # Track signals to avoid spam
        self.last_signal_time = defaultdict(float)
        self.signal_cooldown = 900  # 15 minutes

    async def fetch(self, session, endpoint, params=None):
        """Generic fetch method with error handling"""
        try:
            async with session.get(f"{self.base_url}{endpoint}", params=params, timeout=5) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
        return None

    async def get_all_symbols(self, session):
        """Get all USDT spot trading pairs"""
        data = await self.fetch(session, "/api/v3/exchangeInfo")
        if data:
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['status'] == 'TRADING'
                and s['symbol'].endswith('USDT')
                and s['quoteAsset'] == 'USDT'
                and 'SPOT' in s['permissions']
            ]
            return symbols
        return []

    async def get_ticker(self, session, symbol):
        """Get 24hr ticker data"""
        return await self.fetch(session, "/api/v3/ticker/24hr", {"symbol": symbol})

    async def get_recent_trades(self, session, symbol):
        """Get recent trades"""
        return await self.fetch(session, "/api/v3/trades", {"symbol": symbol, "limit": 100})

    async def get_orderbook(self, session, symbol):
        """Get order book depth"""
        return await self.fetch(session, "/api/v3/depth", {"symbol": symbol, "limit": 20})

    async def get_klines(self, session, symbol, interval="1m", limit=10):
        """Get candlestick data"""
        return await self.fetch(session, "/api/v3/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })

    def analyze_volume_surge(self, symbol, current_volume):
        """Detect abnormal volume spikes"""
        hist = self.volume_history[symbol]
        hist.append(float(current_volume))

        if len(hist) < 10:
            return 0

        recent_avg = mean(list(hist)[-10:])
        historical_avg = mean(list(hist)[:-10]) if len(hist) > 10 else recent_avg

        if historical_avg > 0:
            surge_ratio = recent_avg / historical_avg
            return surge_ratio
        return 0

    def analyze_price_momentum(self, symbol, current_price):
        """Track price acceleration"""
        hist = self.price_history[symbol]
        hist.append(float(current_price))

        if len(hist) < 20:
            return 0, 0

        prices = list(hist)
        recent_change = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
        older_change = (prices[-10] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0

        acceleration = recent_change - older_change
        return recent_change, acceleration

    def analyze_trade_pattern(self, trades):
        """Analyze recent trade patterns for buying pressure"""
        if not trades:
            return 0, 0

        buy_volume = sum(float(t['qty']) for t in trades if not t['isBuyerMaker'])
        sell_volume = sum(float(t['qty']) for t in trades if t['isBuyerMaker'])
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0, 0

        buy_ratio = buy_volume / total_volume

        # Large trade detection
        quantities = [float(t['qty']) for t in trades]
        avg_qty = mean(quantities)
        large_trades = sum(1 for q in quantities if q > avg_qty * 3)
        large_trade_ratio = large_trades / len(trades)

        return buy_ratio, large_trade_ratio

    def analyze_orderbook_imbalance(self, orderbook):
        """Detect bid/ask imbalance indicating accumulation"""
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return 0

        bid_volume = sum(float(b[1]) for b in orderbook['bids'][:10])
        ask_volume = sum(float(a[1]) for a in orderbook['asks'][:10])

        if ask_volume == 0:
            return 0

        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return imbalance

    def analyze_price_levels(self, klines):
        """Detect breakout from consolidation"""
        if not klines or len(klines) < 10:
            return 0, 0

        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]

        # Volatility
        price_range = max(highs) - min(lows)
        avg_price = mean(closes)
        volatility = (price_range / avg_price) if avg_price > 0 else 0

        # Breakout detection
        recent_high = max(highs[-3:])
        historical_resistance = max(highs[:-3])
        breakout = (recent_high - historical_resistance) / historical_resistance if historical_resistance > 0 else 0

        return volatility, breakout

    def calculate_pump_score(self, metrics):
        """Calculate composite pump signal score"""
        score = 0
        signals = []

        # Volume surge (0-30 points)
        if metrics['volume_surge'] > 3:
            score += min(30, metrics['volume_surge'] * 5)
            signals.append(f"Volume Surge: {metrics['volume_surge']:.2f}x")

        # Price acceleration (0-25 points)
        if metrics['price_acceleration'] > 0.01:
            score += min(25, metrics['price_acceleration'] * 1000)
            signals.append(f"Price Acceleration: {metrics['price_acceleration']*100:.2f}%")

        # Buy pressure (0-20 points)
        if metrics['buy_ratio'] > 0.6:
            score += (metrics['buy_ratio'] - 0.5) * 40
            signals.append(f"Buy Pressure: {metrics['buy_ratio']*100:.1f}%")

        # Order book imbalance (0-15 points)
        if metrics['orderbook_imbalance'] > 0.2:
            score += metrics['orderbook_imbalance'] * 75
            signals.append(f"Bid/Ask Imbalance: {metrics['orderbook_imbalance']*100:.1f}%")

        # Large trades (0-10 points)
        if metrics['large_trade_ratio'] > 0.15:
            score += metrics['large_trade_ratio'] * 50
            signals.append(f"Large Trades: {metrics['large_trade_ratio']*100:.1f}%")

        # Breakout bonus (0-10 points)
        if metrics['breakout'] > 0.02:
            score += min(10, metrics['breakout'] * 200)
            signals.append(f"Breakout: {metrics['breakout']*100:.2f}%")

        return score, signals

    async def send_telegram_alert(self, symbol, score, signals, price, volume_24h):
        """Send alert to Telegram"""
        message = f"🚀 <b>PUMP ALERT</b> 🚀\n\n"
        message += f"<b>Symbol:</b> {symbol}\n"
        message += f"<b>Score:</b> {score:.1f}/100\n"
        message += f"<b>Price:</b> ${float(price):.8f}\n"
        message += f"<b>24h Volume:</b> ${float(volume_24h):,.0f}\n\n"
        message += f"<b>Signals Detected:</b>\n"
        for signal in signals:
            message += f"• {signal}\n"
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5) as response:
                    if response.status == 200:
                        print(f"✅ Alert sent for {symbol}")
                    else:
                        print(f"❌ Failed to send alert: {response.status}")
        except Exception as e:
            print(f"❌ Telegram error: {e}")

    async def analyze_symbol(self, session, symbol):
        """Comprehensive analysis of a single symbol"""
        try:
            # Fetch all required data concurrently
            ticker_task = self.get_ticker(session, symbol)
            trades_task = self.get_recent_trades(session, symbol)
            orderbook_task = self.get_orderbook(session, symbol)
            klines_task = self.get_klines(session, symbol)

            ticker, trades, orderbook, klines = await asyncio.gather(
                ticker_task, trades_task, orderbook_task, klines_task,
                return_exceptions=True
            )

            if not ticker or isinstance(ticker, Exception):
                return None

            # Extract metrics
            current_price = float(ticker['lastPrice'])
            volume_24h = float(ticker['volume'])
            quote_volume = float(ticker['quoteVolume'])

            # Skip low volume coins
            if quote_volume < 100000:  # Min $100k daily volume
                return None

            # Calculate all metrics
            volume_surge = self.analyze_volume_surge(symbol, volume_24h)
            price_change, price_acceleration = self.analyze_price_momentum(symbol, current_price)

            buy_ratio, large_trade_ratio = 0, 0
            if trades and not isinstance(trades, Exception):
                buy_ratio, large_trade_ratio = self.analyze_trade_pattern(trades)

            orderbook_imbalance = 0
            if orderbook and not isinstance(orderbook, Exception):
                orderbook_imbalance = self.analyze_orderbook_imbalance(orderbook)

            volatility, breakout = 0, 0
            if klines and not isinstance(klines, Exception):
                volatility, breakout = self.analyze_price_levels(klines)

            metrics = {
                'volume_surge': volume_surge,
                'price_acceleration': price_acceleration,
                'buy_ratio': buy_ratio,
                'orderbook_imbalance': orderbook_imbalance,
                'large_trade_ratio': large_trade_ratio,
                'breakout': breakout,
                'volatility': volatility
            }

            score, signals = self.calculate_pump_score(metrics)

            # Trigger alert if score is high enough
            if score >= 60:  # Threshold for pump signal
                current_time = time.time()
                if current_time - self.last_signal_time[symbol] > self.signal_cooldown:
                    self.last_signal_time[symbol] = current_time
                    await self.send_telegram_alert(symbol, score, signals, current_price, quote_volume)
                    return {"symbol": symbol, "score": score, "price": current_price}

            return None

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

    async def scan_market(self):
        """Main scanning loop"""
        print("🔍 Starting Binance Pump Detector...")
        print(f"📱 Telegram alerts enabled for chat: {self.telegram_chat_id}")

        async with aiohttp.ClientSession() as session:
            # Get all trading pairs
            symbols = await self.get_all_symbols(session)
            print(f"📊 Monitoring {len(symbols)} symbols")

            while True:
                try:
                    start_time = time.time()

                    # Analyze all symbols concurrently in batches
                    batch_size = 50
                    for i in range(0, len(symbols), batch_size):
                        batch = symbols[i:i+batch_size]
                        tasks = [self.analyze_symbol(session, symbol) for symbol in batch]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Log any detected pumps
                        for result in results:
                            if result and not isinstance(result, Exception):
                                print(f"🎯 PUMP DETECTED: {result['symbol']} - Score: {result['score']:.1f}")

                    elapsed = time.time() - start_time
                    print(f"⏱️  Scan completed in {elapsed:.2f}s")

                    # Wait before next scan (adjust based on rate limits)
                    await asyncio.sleep(max(2, 10 - elapsed))

                except Exception as e:
                    print(f"❌ Scan error: {e}")
                    await asyncio.sleep(10)

async def main():
    # Configuration
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("⚠️  Warning: Please set TELEGRAM_BOT_TOKEN environment variable")
        print("You can get a bot token from @BotFather on Telegram")

    if TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
        print("⚠️  Warning: Please set TELEGRAM_CHAT_ID environment variable")
        print("You can get your chat ID from @userinfobot on Telegram")

    detector = BinancePumpDetector(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    await detector.scan_market()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
