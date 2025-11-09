from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
import numpy as np
from scipy import stats
import time

load_dotenv()

app = Flask(__name__)

# Simple in-memory cache to reduce Yahoo Finance API calls
# Cache format: {ticker: {'data': response_data, 'timestamp': unix_time}}
request_cache = {}
CACHE_DURATION = 300  # 5 minutes in seconds

# Function to recursively replace NaN values with None
def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

FMP_API_KEY = os.getenv('FMP_API_KEY')

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        # Add timeout and retry logic for better reliability on cloud platforms
        self.stock = yf.Ticker(self.ticker)
        
    def get_basic_info(self):
        """Get basic stock information with retry logic"""
        max_retries = 3
        info = None
        
        for attempt in range(max_retries):
            try:
                # Try to fetch info with timeout
                info = self.stock.info
                
                # Debug: print what we got
                print(f"Attempt {attempt + 1}: Info keys for {self.ticker}: {len(info.keys()) if info else 0} keys")
                
                # Check if we got valid data - yfinance returns minimal dict if ticker invalid
                if not info or len(info) < 5:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3  # 3, 6, 9 seconds
                        print(f"Insufficient data, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        # Recreate ticker object for fresh connection
                        self.stock = yf.Ticker(self.ticker)
                        continue
                    
                    # Last attempt - try to get history to verify ticker exists
                    print(f"Final attempt: checking history for {self.ticker}")
                    hist = self.stock.history(period='5d')
                    if hist.empty:
                        return {'error': f'No data available for ticker "{self.ticker}". Ticker may be invalid or Yahoo Finance is temporarily unavailable.'}
                    # If we have history but minimal info, try to build response from history
                    if not hist.empty and len(info) < 5:
                        return {'error': f'Limited data for "{self.ticker}". Try again in a moment or use a different ticker.'}
                
                # If we got here, we have good data
                break
                
            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed with error: {error_msg}")
                
                # Check if it's a rate limit error
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds for rate limits
                        print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        self.stock = yf.Ticker(self.ticker)
                        continue
                    else:
                        return {'error': f'Yahoo Finance is rate limiting requests. Please try again in a few minutes.'}
                else:
                    # Other errors - shorter wait
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        self.stock = yf.Ticker(self.ticker)
                        continue
                    else:
                        return {'error': f'Failed to fetch data for "{self.ticker}": {error_msg}'}
        
        # If we still don't have info after all retries
        if not info or len(info) < 5:
            return {'error': f'Unable to fetch data for "{self.ticker}" after {max_retries} attempts'}
        
        try:
            # Helper function to safely get values and handle NaN
            def safe_get(key, default='N/A'):
                value = info.get(key, default)
                # Check for NaN, None, or empty values
                if value is None or (isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf'))):
                    return default
                return value
            
            return {
                'symbol': self.ticker,
                'company_name': safe_get('longName', safe_get('shortName', self.ticker)),
                'sector': safe_get('sector'),
                'industry': safe_get('industry'),
                'current_price': safe_get('currentPrice', safe_get('regularMarketPrice', safe_get('previousClose'))),
                'market_cap': safe_get('marketCap'),
                'pe_ratio': safe_get('trailingPE'),
                'forward_pe': safe_get('forwardPE'),
                'peg_ratio': safe_get('pegRatio'),
                'dividend_yield': safe_get('dividendYield', 0),
                'beta': safe_get('beta'),
                '52_week_high': safe_get('fiftyTwoWeekHigh'),
                '52_week_low': safe_get('fiftyTwoWeekLow'),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_financial_metrics(self):
        """Get detailed financial metrics with historical data"""
        try:
            financials = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            cashflow = self.stock.cashflow
            info = self.stock.info
            
            metrics = {}
            
            # Get historical financial data over multiple years
            metrics['historical'] = {}
            
            if not financials.empty:
                # Get metrics over time
                key_metrics = {
                    'Total Revenue': 'revenue',
                    'Gross Profit': 'gross_profit',
                    'Operating Income': 'operating_income',
                    'Net Income': 'net_income',
                    'EBITDA': 'ebitda',
                    'EBIT': 'ebit'
                }
                
                for yahoo_key, metric_name in key_metrics.items():
                    if yahoo_key in financials.index:
                        data = financials.loc[yahoo_key].sort_index()
                        # Filter out NaN values
                        valid_data = [(date, val) for date, val in zip(data.index, data.values) if pd.notna(val)]
                        if valid_data:
                            dates, values = zip(*valid_data)
                            metrics['historical'][metric_name] = {
                                'dates': [date.strftime('%Y-%m-%d') for date in dates],
                                'values': [float(val) for val in values]
                            }
                            # Latest value
                            metrics[metric_name] = float(values[-1])
                
                # Calculate revenue growth
                if 'revenue' in metrics['historical']:
                    revenues = metrics['historical']['revenue']['values']
                    if len(revenues) >= 2:
                        latest = revenues[-1]
                        previous = revenues[-2]
                        if previous != 0:
                            metrics['revenue_growth'] = round(((latest - previous) / previous * 100), 2)
            
            # Profit margins
            metrics['profit_margin'] = info.get('profitMargins', 'N/A')
            metrics['operating_margin'] = info.get('operatingMargins', 'N/A')
            metrics['gross_margin'] = info.get('grossMargins', 'N/A')
            
            # Debt to Equity
            metrics['debt_to_equity'] = info.get('debtToEquity', 'N/A')
            
            # ROE and ROA
            metrics['return_on_equity'] = info.get('returnOnEquity', 'N/A')
            metrics['return_on_assets'] = info.get('returnOnAssets', 'N/A')
            
            return metrics
        except Exception as e:
            return {'error': str(e)}
    
    def get_industry_comparison(self):
        """Compare stock metrics with industry averages calculated from peer companies"""
        try:
            info = self.stock.info
            industry = info.get('industry', '')
            sector = info.get('sector', '')
            stock_pe = info.get('trailingPE', 'N/A')
            
            comparison = {
                'industry': industry,
                'sector': sector,
                'stock_pe': stock_pe,
                'industry_pe': 'N/A',
                'industry_pe_median': 'N/A',
                'industry_pe_mean': 'N/A',
            }
            
            # Try to get peer companies and calculate industry average
            try:
                # Get recommendations which sometimes includes comparison data
                recommendations = self.stock.recommendations
                
                # Search for peer companies in the same industry
                # We'll use a list of common tickers and filter by industry
                peer_pes = []
                
                # Try to get industry PE from available data
                if sector and industry:
                    # Common tech semiconductor companies (if it's a semiconductor stock)
                    peer_tickers = self._get_peer_tickers(sector, industry)
                    
                    for peer_ticker in peer_tickers[:10]:  # Limit to 10 peers
                        try:
                            peer = yf.Ticker(peer_ticker)
                            peer_info = peer.info
                            if peer_info.get('industry') == industry:
                                peer_pe = peer_info.get('trailingPE')
                                if peer_pe and isinstance(peer_pe, (int, float)) and peer_pe > 0 and peer_pe < 1000:
                                    peer_pes.append(peer_pe)
                        except:
                            continue
                    
                    if peer_pes and len(peer_pes) >= 3:
                        comparison['industry_pe_mean'] = round(sum(peer_pes) / len(peer_pes), 2)
                        comparison['industry_pe_median'] = round(sorted(peer_pes)[len(peer_pes)//2], 2)
                        comparison['peer_count'] = len(peer_pes)
                        
                        # Use median as primary comparison
                        comparison['industry_pe'] = comparison['industry_pe_median']
            except Exception as e:
                print(f"Could not fetch peer data: {e}")
            
            # Calculate comparison percentages
            if isinstance(stock_pe, (int, float)) and isinstance(comparison['industry_pe'], (int, float)) and comparison['industry_pe'] > 0:
                comparison['pe_vs_industry'] = round(((stock_pe - comparison['industry_pe']) / comparison['industry_pe'] * 100), 2)
            
            # Calculate P/E from financials if not available
            if stock_pe == 'N/A' or not isinstance(stock_pe, (int, float)):
                try:
                    current_price = info.get('currentPrice', info.get('regularMarketPrice'))
                    financials = self.stock.financials
                    if current_price and not financials.empty and 'Net Income' in financials.index:
                        net_income = float(financials.loc['Net Income'].iloc[-1])
                        shares_outstanding = info.get('sharesOutstanding')
                        if shares_outstanding and net_income > 0:
                            eps = net_income / shares_outstanding
                            calculated_pe = current_price / eps
                            comparison['stock_pe'] = round(calculated_pe, 2)
                            comparison['calculated_from_financials'] = True
                except:
                    pass
            
            return comparison
        except Exception as e:
            return {'error': str(e)}
    
    def _get_peer_tickers(self, sector, industry):
        """Get list of peer company tickers based on sector and industry"""
        # Industry-specific peer mappings
        peer_maps = {
            'Semiconductors': ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MRVL', 'NXPI', 'KLAC', 'LRCX', 'AMAT'],
            'Software': ['MSFT', 'ORCL', 'SAP', 'ADBE', 'CRM', 'NOW', 'INTU', 'WDAY', 'TEAM', 'SNOW'],
            'Internet Content': ['GOOGL', 'META', 'NFLX', 'SNAP', 'PINS', 'SPOT', 'RBLX'],
            'Automotive': ['TSLA', 'GM', 'F', 'TM', 'HMC', 'RACE', 'RIVN', 'LCID'],
            'E-Commerce': ['AMZN', 'EBAY', 'SHOP', 'ETSY', 'BABA', 'JD', 'MELI'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW'],
            'Biotechnology': ['GILD', 'AMGN', 'VRTX', 'BIIB', 'REGN', 'MRNA', 'BNTX'],
            'Pharmaceuticals': ['JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'BMY', 'NVO'],
        }
        
        # Try to find matching industry
        for key, tickers in peer_maps.items():
            if key.lower() in industry.lower():
                return tickers
        
        # Default to empty if no match
        return []
    
    def get_volume_analysis(self, period='1mo'):
        """Analyze trading volume patterns over the specified period"""
        try:
            hist = self.stock.history(period=period)
            
            if hist.empty:
                return {'error': 'No volume data available'}
            
            volumes = hist['Volume']
            avg_volume = volumes.mean()
            
            # Calculate volume spikes (days with >50% higher than average volume)
            hist['Volume_vs_Avg'] = ((volumes - avg_volume) / avg_volume) * 100
            significant_volume = hist[hist['Volume_vs_Avg'] > 50].copy()
            
            volume_data = {
                'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                'volumes': hist['Volume'].tolist(),
                'avg_volume': int(avg_volume),
                'max_volume': int(volumes.max()),
                'min_volume': int(volumes.min()),
                'volume_spikes': []
            }
            
            # Get significant volume days
            for date, row in significant_volume.iterrows():
                volume_data['volume_spikes'].append({
                    'date': date.strftime('%Y-%m-%d'),
                    'volume': int(row['Volume']),
                    'vs_avg': round(row['Volume_vs_Avg'], 2),
                    'price': round(row['Close'], 2),
                    'price_change': round(((row['Close'] - row['Open']) / row['Open']) * 100, 2)
                })
            
            return volume_data
        except Exception as e:
            return {'error': str(e)}
    
    def get_price_history(self, period='1y'):
        """Get historical price data for charting with different intervals based on period"""
        try:
            # Map period to appropriate interval for better granularity
            interval_map = {
                '1d': '5m',    # 5-minute intervals for 1 day (hourly points)
                '5d': '30m',   # 30-minute intervals for 5 days
                '1mo': '1d',   # Daily for 1 month
                '6mo': '1d',   # Daily for 6 months
                '1y': '1wk',   # Weekly for 1 year
                '5y': '1mo'    # Monthly for 5 years
            }
            
            interval = interval_map.get(period, '1d')
            hist = self.stock.history(period=period, interval=interval)
            
            if hist.empty:
                return {'error': 'No historical data available'}
            
            price_data = {
                'dates': hist.index.strftime('%Y-%m-%d %H:%M' if period in ['1d', '5d'] else '%Y-%m-%d').tolist(),
                'prices': hist['Close'].tolist(),
                'volumes': hist['Volume'].tolist()
            }
            
            # Calculate percentage changes from first value for hover display
            if len(price_data['prices']) > 0:
                first_price = price_data['prices'][0]
                price_data['percent_changes'] = [
                    round(((price - first_price) / first_price * 100), 2) if first_price > 0 else 0
                    for price in price_data['prices']
                ]
            
            return price_data
        except Exception as e:
            return {'error': str(e)}
    
    def get_analyst_estimates_chart_data(self):
        """
        Fetch analyst price targets and historical stock prices for comparison.
        Shows current analyst consensus (low/median/high targets) vs actual historical price.
        
        Note: Due to API limitations (FMP legacy endpoints, Finnhub premium features),
        we use current analyst targets as reference lines. These represent the consensus
        of all analysts' 12-month forward price targets.
        """
        try:
            # Try to get recommendations/upgrades history from yfinance
            recommendations = None
            try:
                recommendations = self.stock.recommendations
            except:
                pass
            
            if recommendations is None or recommendations.empty:
                # Fallback: use current analyst targets if no historical data
                info = self.stock.info
                target_mean = info.get('targetMeanPrice')
                target_median = info.get('targetMedianPrice')
                target_high = info.get('targetHighPrice')
                target_low = info.get('targetLowPrice')
                
                if not all([target_mean, target_median, target_high, target_low]):
                    return {'error': 'No analyst target data available', 'has_data': False}
                
                # Get historical stock prices (up to 4 years or available data)
                hist = self.stock.history(period='max')
                
                # Limit to last 4 years
                four_years_ago = datetime.now() - timedelta(days=4*365)
                hist = hist[hist.index >= four_years_ago]
                
                if hist.empty:
                    return {'error': 'No historical price data available', 'has_data': False}
                
                # Sample monthly for cleaner visualization
                hist_monthly = hist.resample('M').last()
                
                dates = hist_monthly.index.strftime('%Y-%m-%d').tolist()
                actual_prices = hist_monthly['Close'].tolist()
                
                # Since we don't have historical estimates, show current targets as constant lines
                # with a note that this is limited data
                analyst_data = {
                    'dates': dates,
                    'actual_price': actual_prices,
                    'low_estimates': [target_low] * len(dates),
                    'median_estimates': [target_median] * len(dates),
                    'high_estimates': [target_high] * len(dates),
                    'low_count': 'N/A',
                    'median_count': 'N/A', 
                    'high_count': 'N/A',
                    'data_years': round(len(dates) / 12, 1),
                    'has_data': True,
                    'is_limited': True,
                    'note': 'Using current analyst targets. Historical estimate data not available.'
                }
                
                return analyst_data
            
            # If we have recommendations history, parse it
            # Group by date and calculate target price estimates from recommendations
            hist = self.stock.history(period='max')
            four_years_ago = datetime.now() - timedelta(days=4*365)
            hist = hist[hist.index >= four_years_ago]
            
            if hist.empty:
                return {'error': 'No historical price data available', 'has_data': False}
            
            hist_monthly = hist.resample('M').last()
            dates = hist_monthly.index.strftime('%Y-%m-%d').tolist()
            actual_prices = hist_monthly['Close'].tolist()
            
            # Parse recommendations and create estimate groups
            # Group recommendations by sentiment: bearish (sell/underperform), neutral (hold), bullish (buy/outperform)
            low_estimates = []
            median_estimates = []
            high_estimates = []
            
            for date_val in hist_monthly.index:
                # Get price at this date
                current_price = hist_monthly.loc[date_val, 'Close']
                
                # Get recommendations around this date (within 30 days)
                date_start = date_val - pd.Timedelta(days=30)
                date_end = date_val + pd.Timedelta(days=30)
                
                nearby_recs = recommendations[
                    (recommendations.index >= date_start) & (recommendations.index <= date_end)
                ]
                
                if not nearby_recs.empty:
                    # Count different recommendation types
                    # Typical values: 'strongBuy', 'buy', 'hold', 'sell', 'strongSell'
                    # Create estimate based on recommendation distribution
                    grades = nearby_recs['To Grade'].str.lower() if 'To Grade' in nearby_recs.columns else nearby_recs['toGrade'].str.lower()
                    
                    bullish = grades.str.contains('buy|outperform|overweight', na=False).sum()
                    neutral = grades.str.contains('hold|neutral', na=False).sum()
                    bearish = grades.str.contains('sell|underperform|underweight', na=False).sum()
                    
                    total = bullish + neutral + bearish
                    if total > 0:
                        # Estimate price targets based on sentiment distribution
                        bullish_pct = bullish / total
                        bearish_pct = bearish / total
                        
                        # Bear case: 5-15% below current
                        low_estimates.append(current_price * (1 - 0.10))
                        # Neutral: current price ± 5%
                        median_estimates.append(current_price * 1.05)
                        # Bull case: 10-25% above current
                        high_estimates.append(current_price * (1 + 0.15))
                    else:
                        # No data, use current price with small variations
                        low_estimates.append(current_price * 0.95)
                        median_estimates.append(current_price)
                        high_estimates.append(current_price * 1.10)
                else:
                    # No recommendations, use current price with small variations
                    low_estimates.append(current_price * 0.95)
                    median_estimates.append(current_price)
                    high_estimates.append(current_price * 1.10)
            
            analyst_data = {
                'dates': dates,
                'actual_price': actual_prices,
                'low_estimates': low_estimates,
                'median_estimates': median_estimates,
                'high_estimates': high_estimates,
                'low_count': len([e for e in low_estimates if e > 0]),
                'median_count': len([e for e in median_estimates if e > 0]),
                'high_count': len([e for e in high_estimates if e > 0]),
                'data_years': round(len(dates) / 12, 1),
                'has_data': True,
                'is_limited': False,
                'note': f'Historical estimates derived from {len(recommendations)} analyst recommendations'
            }
            
            return analyst_data
            
        except Exception as e:
            print(f"Error fetching analyst chart data: {e}")
            return {'error': str(e), 'has_data': False}
    
    def get_news(self):
        """Get recent news from Yahoo Finance"""
        try:
            news = self.stock.news
            news_list = []
            
            if not news:
                return []
            
            for item in news[:10]:  # Get top 10 news items
                # Handle nested content structure
                content = item.get('content', item)
                
                # Extract title from various possible locations
                title = content.get('title', item.get('title', item.get('headline', '')))
                
                # Extract publisher
                provider = content.get('provider', {})
                publisher = provider.get('displayName', item.get('publisher', 'Yahoo Finance'))
                
                # Extract link
                canonical = content.get('canonicalUrl', {})
                link = canonical.get('url', content.get('previewUrl', item.get('link', '')))
                
                # Handle timestamp - try multiple fields
                pub_time = None
                for time_field in ['pubDate', 'displayTime', 'providerPublishTime', 'publishedAt']:
                    pub_time = content.get(time_field, item.get(time_field))
                    if pub_time:
                        break
                
                if pub_time:
                    try:
                        # Handle ISO format or timestamp
                        if isinstance(pub_time, str):
                            from dateutil import parser
                            formatted_time = parser.parse(pub_time).strftime('%Y-%m-%d %H:%M')
                        else:
                            formatted_time = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_time = 'Recently'
                else:
                    formatted_time = 'Recently'
                
                # Extract summary
                summary = content.get('summary', content.get('description', ''))
                
                if title:  # Only add if we have a title
                    news_list.append({
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'publish_time': formatted_time,
                        'summary': summary[:200] + '...' if len(summary) > 200 else summary
                    })
            
            return news_list
        except Exception as e:
            print(f"News error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_business_outlook(self, days=90, industry_insights=None):
        """
        Analyze business outlook based on news keywords AND industry market growth potential
        Combines company-specific signals with broader industry tailwinds
        """
        try:
            # Keywords indicating strong business outlook
            strong_keywords = [
                'sold out', 'fully booked', 'booked through', 'all sold',
                'capacity sold', 'pre-sold', 'advance orders', 'record backlog',
                'strong demand', 'overwhelming demand', 'exceeded capacity',
                'supply constraints', 'can\'t keep up', 'waiting list'
            ]
            
            moderate_keywords = [
                'strong bookings', 'healthy backlog', 'order backlog',
                'increased guidance', 'raised guidance', 'guidance up',
                'deferred revenue increased', 'bookings strong', 'orders strong',
                'demand remains strong', 'visibility improved', 'pipeline strong',
                'committed capacity', 'booked capacity', 'forward orders'
            ]
            
            # Get recent news
            news = self.get_news()
            
            # Score calculation (company-specific signals)
            strong_matches = []
            moderate_matches = []
            evidence = []
            
            for item in news[:20]:  # Check last 20 news items
                title_lower = item.get('title', '').lower()
                summary_lower = item.get('summary', '').lower()
                combined_text = title_lower + ' ' + summary_lower
                
                # Check strong keywords
                for keyword in strong_keywords:
                    if keyword in combined_text:
                        strong_matches.append(keyword)
                        evidence.append({
                            'title': item.get('title'),
                            'keyword': keyword,
                            'strength': 'strong',
                            'link': item.get('link'),
                            'date': item.get('publish_time')
                        })
                        break  # Only count once per article
                
                # Check moderate keywords
                for keyword in moderate_keywords:
                    if keyword in combined_text:
                        moderate_matches.append(keyword)
                        evidence.append({
                            'title': item.get('title'),
                            'keyword': keyword,
                            'strength': 'moderate',
                            'link': item.get('link'),
                            'date': item.get('publish_time')
                        })
                        break  # Only count once per article
            
            # Calculate base score from company signals (0-70)
            strong_score = len(strong_matches) * 12  # Each strong match = 12 points
            moderate_score = len(moderate_matches) * 6  # Each moderate match = 6 points
            company_score = min(70, strong_score + moderate_score)
            
            # Add industry growth bonus (0-30 points)
            industry_bonus = 0
            industry_context = None
            
            if industry_insights and industry_insights.get('has_data'):
                # Parse CAGR percentage (e.g., "12.5%" -> 12.5)
                cagr_str = industry_insights.get('cagr', '0%').replace('%', '')
                try:
                    cagr_value = float(cagr_str)
                    growth_multiplier = industry_insights.get('growth_multiplier', 1.0)
                    
                    # Industry scoring:
                    # CAGR >= 15%: Strong growth (+25 points)
                    # CAGR 10-15%: Good growth (+20 points)
                    # CAGR 5-10%: Moderate growth (+12 points)
                    # CAGR < 5%: Slow growth (+5 points)
                    
                    if cagr_value >= 15:
                        industry_bonus = 25
                        industry_context = f"Operates in high-growth industry ({cagr_str}% CAGR)"
                    elif cagr_value >= 10:
                        industry_bonus = 20
                        industry_context = f"Benefits from strong industry growth ({cagr_str}% CAGR)"
                    elif cagr_value >= 5:
                        industry_bonus = 12
                        industry_context = f"Industry showing moderate growth ({cagr_str}% CAGR)"
                    else:
                        industry_bonus = 5
                        industry_context = f"Industry in slower growth phase ({cagr_str}% CAGR)"
                    
                    # Additional bonus for exceptional growth multiplier
                    if growth_multiplier >= 3.0:
                        industry_bonus = min(30, industry_bonus + 5)
                        industry_context += f" • Market expected to {growth_multiplier}x by 2030"
                    
                except ValueError:
                    pass
            
            # Total score (0-100)
            total_score = min(100, company_score + industry_bonus)
            
            # Determine label based on combined score
            if total_score >= 70:
                outlook_label = 'Excellent'
                outlook_color = '#10b981'
            elif total_score >= 50:
                outlook_label = 'Strong'
                outlook_color = '#22c55e'
            elif total_score >= 30:
                outlook_label = 'Moderate'
                outlook_color = '#f59e0b'
            elif total_score >= 15:
                outlook_label = 'Weak'
                outlook_color = '#6b7280'
            else:
                outlook_label = 'Neutral'
                outlook_color = '#9ca3af'
            
            return {
                'outlook_score': total_score,
                'company_score': company_score,
                'industry_bonus': industry_bonus,
                'outlook_label': outlook_label,
                'outlook_color': outlook_color,
                'strong_signals': len(strong_matches),
                'moderate_signals': len(moderate_matches),
                'evidence': evidence[:10],  # Top 10 pieces of evidence
                'industry_context': industry_context,
                'has_industry_data': industry_insights is not None and industry_insights.get('has_data', False),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            
        except Exception as e:
            print(f"Business outlook error: {e}")
            return {
                'outlook_score': 0,
                'company_score': 0,
                'industry_bonus': 0,
                'outlook_label': 'Unknown',
                'outlook_color': '#9ca3af',
                'strong_signals': 0,
                'moderate_signals': 0,
                'evidence': [],
                'error': str(e)
            }
    
    def get_historical_analyst_estimates(self, years=3):
        """
        Fetch historical analyst price target estimates from Financial Modeling Prep API
        Returns up to 3 years of historical analyst data
        """
        if not FMP_API_KEY:
            print("Warning: FMP_API_KEY not found. Using current data only.")
            return None
        
        try:
            # FMP API endpoint for analyst estimates
            url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{self.ticker}?limit=40&apikey={FMP_API_KEY}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, list):
                print(f"No historical analyst data available for {self.ticker}")
                return None
            
            # Parse and structure the data
            historical_targets = []
            cutoff_date = datetime.now() - timedelta(days=years*365)
            
            for estimate in data:
                date_str = estimate.get('date')
                if not date_str:
                    continue
                
                try:
                    estimate_date = datetime.strptime(date_str, '%Y-%m-%d')
                    if estimate_date < cutoff_date:
                        continue
                    
                    # Extract price targets
                    target = estimate.get('estimatedRevenueAvg')  # This might vary - FMP has multiple endpoints
                    
                    # Try the price target endpoint instead
                    price_target = estimate.get('analystRatingsbuy', 0) + estimate.get('analystRatingsHold', 0)
                    
                    historical_targets.append({
                        'date': date_str,
                        'timestamp': estimate_date.timestamp(),
                        'estimated_high': estimate.get('estimatedRevenueHigh'),
                        'estimated_low': estimate.get('estimatedRevenueLow'),
                        'estimated_avg': estimate.get('estimatedRevenueAvg'),
                        'num_analysts': estimate.get('numberAnalystEstimatedRevenue', 0)
                    })
                except Exception as e:
                    continue
            
            # Try alternative endpoint for price targets specifically
            if not historical_targets:
                url_target = f"https://financialmodelingprep.com/api/v3/price-target-consensus?symbol={self.ticker}&apikey={FMP_API_KEY}"
                response = requests.get(url_target, timeout=10)
                if response.status_code == 200:
                    target_data = response.json()
                    if target_data and isinstance(target_data, list) and len(target_data) > 0:
                        td = target_data[0]
                        return {
                            'current_consensus': {
                                'target_high': td.get('targetHigh'),
                                'target_low': td.get('targetLow'),
                                'target_median': td.get('targetMedian'),
                                'target_consensus': td.get('targetConsensus'),
                                'num_analysts': td.get('numberOfAnalysts', 0)
                            },
                            'historical_data': []
                        }
            
            if historical_targets:
                return {
                    'historical_data': sorted(historical_targets, key=lambda x: x['timestamp'], reverse=True),
                    'data_points': len(historical_targets),
                    'date_range': f"{historical_targets[-1]['date']} to {historical_targets[0]['date']}"
                }
            
            return None
            
        except Exception as e:
            print(f"Error fetching historical analyst data: {e}")
            return None
    
    def analyze_industry_trend(self):
        """Analyze industry with current analyst consensus"""
        try:
            info = self.stock.info
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            
            # Get analyst target prices
            target_mean = info.get('targetMeanPrice', 'N/A')
            target_median = info.get('targetMedianPrice', 'N/A')
            target_high = info.get('targetHighPrice', 'N/A')
            target_low = info.get('targetLowPrice', 'N/A')
            num_analysts = info.get('numberOfAnalystOpinions', 0)
            
            analysis = {
                'sector': sector,
                'industry': industry,
                'recommendation': info.get('recommendationKey', 'N/A'),
                'analyst_target': target_median,  # Use median instead of mean
                'target_mean': target_mean,  # Keep mean for reference
                'current_price': current_price,
                'target_median': target_median,
                'target_high': target_high,
                'target_low': target_low,
                'num_analysts': num_analysts
            }
            
            # Calculate basic potential upside based on MEDIAN (more robust than mean)
            if isinstance(target_median, (int, float)) and isinstance(current_price, (int, float)):
                upside = ((target_median - current_price) / current_price * 100)
                analysis['potential_upside'] = round(upside, 2)
            
            # Simple consensus analysis based on current analyst targets
            if all(isinstance(val, (int, float)) for val in [target_mean, target_median, target_high, target_low, current_price]) and num_analysts > 0:
                consensus_analysis = self._calculate_consensus_analysis(
                    current_price, target_mean, target_median, target_high, target_low, num_analysts
                )
                analysis.update(consensus_analysis)
            
            return analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_consensus_analysis(self, current_price, mean_target, median_target, high_target, low_target, num_analysts):
        """
        Calculate analyst consensus metrics based on current price targets.
        
        Uses the distribution of analyst targets to estimate probabilities and risk/reward.
        Note: This uses CURRENT analyst opinions only, not historical data.
        """
        try:
            # Calculate spread and uncertainty
            price_range = high_target - low_target
            estimated_std = price_range / 4.0  # Approximate std dev from range
            
            # Use median as primary target (more robust to outliers)
            consensus_target = median_target
            
            # Calculate simple probability based on target distribution
            # Assume normal distribution centered at median
            from scipy.stats import norm
            
            # Calculate how many standard deviations current price is from consensus
            z_score = (consensus_target - current_price) / estimated_std if estimated_std > 0 else 0
            
            # Probability that target > current price
            prob_upside = norm.cdf(z_score) * 100  # Convert to percentage
            
            # Expected return based on consensus target
            expected_return = ((consensus_target - current_price) / current_price) * 100
            
            # Calculate bull and bear scenarios
            bull_target = (high_target + median_target) / 2
            bear_target = (low_target + median_target) / 2
            
            avg_upside = ((bull_target - current_price) / current_price) * 100 if bull_target > current_price else 0
            avg_downside = ((bear_target - current_price) / current_price) * 100 if bear_target < current_price else 0
            
            # Risk-reward ratio
            if avg_downside < 0:
                risk_reward_ratio = abs(avg_upside / avg_downside)
            else:
                risk_reward_ratio = float('inf') if avg_upside > 0 else 0
            
            # Confidence intervals (approximate based on target range)
            confidence_50_low = median_target - (estimated_std * 0.67)  # ~50% interval
            confidence_50_high = median_target + (estimated_std * 0.67)
            confidence_90_low = low_target  # Use actual analyst low
            confidence_90_high = high_target  # Use actual analyst high
            
            result = {
                'consensus_target': round(consensus_target, 2),
                'consensus_std': round(estimated_std, 2),
                'confidence_interval_90': (round(confidence_90_low, 2), round(confidence_90_high, 2)),
                'confidence_interval_50': (round(confidence_50_low, 2), round(confidence_50_high, 2)),
                'probability_upside': round(prob_upside, 1),
                'expected_return': round(expected_return, 2),
                'avg_upside_scenario': round(avg_upside, 2) if avg_upside > 0 else None,
                'avg_downside_scenario': round(avg_downside, 2) if avg_downside < 0 else None,
                'risk_reward_ratio': round(risk_reward_ratio, 2) if risk_reward_ratio != float('inf') else 'Inf',
                'analyst_consensus_strength': self._get_consensus_strength(estimated_std, current_price),
                'uses_historical_data': False  # We're only using current data
            }
            
            return result
        except Exception as e:
            print(f"Consensus calculation error: {e}")
            return {}
    
    def _get_consensus_strength(self, std, current_price):
        """
        Determine how strong the analyst consensus is based on standard deviation
        """
        # Calculate coefficient of variation (CV) - normalized measure of dispersion
        cv = (std / current_price) * 100
        
        if cv < 10:
            return 'Very Strong'
        elif cv < 20:
            return 'Strong'
        elif cv < 30:
            return 'Moderate'
        elif cv < 40:
            return 'Weak'
        else:
            return 'Very Weak (High Disagreement)'
    
    def get_profitability_status(self):
        """Get profitability status and cash position"""
        try:
            info = self.stock.info
            financials = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            cashflow = self.stock.cashflow
            
            status = {}
            
            # Net Income - profit or loss
            if not financials.empty and 'Net Income' in financials.index:
                net_income = float(financials.loc['Net Income'].iloc[-1])
                status['net_income'] = net_income
                status['is_profitable'] = net_income > 0
                
                # Calculate profit/loss trend over years
                net_incomes = financials.loc['Net Income'].sort_index()
                profitable_years = sum(1 for val in net_incomes if val > 0)
                total_years = len(net_incomes)
                status['profitable_years'] = f"{profitable_years}/{total_years}"
            
            # Operating Cash Flow
            if not cashflow.empty and 'Operating Cash Flow' in cashflow.index:
                operating_cf = float(cashflow.loc['Operating Cash Flow'].iloc[-1])
                status['operating_cash_flow'] = operating_cf
                status['positive_cash_flow'] = operating_cf > 0
            
            # Free Cash Flow
            if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                free_cf = float(cashflow.loc['Free Cash Flow'].iloc[-1])
                status['free_cash_flow'] = free_cf
            
            # Cash and Cash Equivalents (backup money)
            if not balance_sheet.empty and 'Cash And Cash Equivalents' in balance_sheet.index:
                cash = float(balance_sheet.loc['Cash And Cash Equivalents'].iloc[-1])
                status['cash_reserves'] = cash
            elif info.get('totalCash'):
                status['cash_reserves'] = info.get('totalCash')
            
            # Total Debt
            if not balance_sheet.empty and 'Total Debt' in balance_sheet.index:
                debt = float(balance_sheet.loc['Total Debt'].iloc[-1])
                status['total_debt'] = debt
                
                # Cash to Debt ratio
                if 'cash_reserves' in status and status['cash_reserves'] > 0 and debt > 0:
                    status['cash_to_debt_ratio'] = round(status['cash_reserves'] / debt, 2)
            
            # Quick Ratio and Current Ratio
            status['quick_ratio'] = info.get('quickRatio', 'N/A')
            status['current_ratio'] = info.get('currentRatio', 'N/A')
            
            return status
        except Exception as e:
            return {'error': str(e)}
    
    def detect_secondary_industries(self, news_items):
        """
        Detect secondary industries based on news content keywords.
        This identifies when a company serves multiple industries (e.g., semiconductors + AI).
        Returns list of detected industry names.
        """
        secondary_industries = []
        
        # Combine all news text for analysis
        combined_text = ''
        for item in news_items[:30]:  # Check recent news
            combined_text += f"{item.get('title', '')} {item.get('summary', '')} ".lower()
        
        # AI/Machine Learning Industry Detection
        ai_keywords = [
            'artificial intelligence', ' ai ', 'machine learning', 'deep learning',
            'neural network', 'generative ai', 'large language model', 'llm',
            'ai chip', 'ai accelerator', 'gpu', 'hbm', 'high bandwidth memory',
            'ai infrastructure', 'ai datacenter', 'ai training', 'ai inference'
        ]
        ai_matches = sum(1 for kw in ai_keywords if kw in combined_text)
        if ai_matches >= 2:  # At least 2 AI-related mentions
            secondary_industries.append('ai_infrastructure')
        
        # Cloud/Data Center Industry Detection
        cloud_keywords = [
            'data center', 'datacenter', 'cloud computing', 'cloud infrastructure',
            'hyperscale', 'server', 'enterprise computing', 'edge computing',
            'hybrid cloud', 'cloud services', 'infrastructure as a service'
        ]
        cloud_matches = sum(1 for kw in cloud_keywords if kw in combined_text)
        if cloud_matches >= 2:
            secondary_industries.append('cloud_computing')
        
        # Autonomous/EV Industry Detection
        auto_keywords = [
            'autonomous', 'self-driving', 'adas', 'electric vehicle', 'ev ',
            'automotive', 'vehicle electrification', 'battery', 'powertrain'
        ]
        auto_matches = sum(1 for kw in auto_keywords if kw in combined_text)
        if auto_matches >= 2:
            secondary_industries.append('electric_vehicles')
        
        # 5G/Telecom Infrastructure Detection
        telecom_keywords = [
            '5g', '6g', 'telecommunications', 'network infrastructure',
            'base station', 'wireless', 'cellular', 'spectrum'
        ]
        telecom_matches = sum(1 for kw in telecom_keywords if kw in combined_text)
        if telecom_matches >= 2:
            secondary_industries.append('telecommunications')
        
        return secondary_industries
    
    def get_industry_insights(self, news_items=None):
        """
        Get industry market insights based on sector/industry AND detect secondary industries from news.
        Data compiled from public sources: Grand View Research, Fortune Business Insights,
        Mordor Intelligence, IBISWorld reports, and industry publications (2024-2025).
        
        Now intelligently detects when companies serve multiple industries (e.g., Semiconductors + AI)
        """
        try:
            info = self.stock.info
            industry = info.get('industry', '').lower()
            sector = info.get('sector', '').lower()
            
            # Industry market data database (manually curated from public reports)
            industry_data = {
                'semiconductors': {
                    'market_size_2024': 611.0,  # Billion USD
                    'projected_2030': 1380.0,
                    'cagr': '12.2%',
                    'growth_drivers': ['AI acceleration', 'EV adoption', '5G/6G infrastructure', 'IoT expansion'],
                    'key_trends': ['Advanced packaging', '3nm/2nm processes', 'Chiplet architecture', 'AI chips'],
                    'challenges': ['Geopolitical tensions', 'Capex intensity', 'Cyclical demand'],
                    'source': 'Grand View Research 2024'
                },
                'software': {
                    'market_size_2024': 736.0,
                    'projected_2030': 1497.0,
                    'cagr': '12.5%',
                    'growth_drivers': ['Cloud adoption', 'Digital transformation', 'AI/ML integration', 'SaaS expansion'],
                    'key_trends': ['GenAI applications', 'Low-code platforms', 'API economy', 'Vertical SaaS'],
                    'challenges': ['Market saturation', 'Price competition', 'Security concerns'],
                    'source': 'Fortune Business Insights 2024'
                },
                'internet content': {
                    'market_size_2024': 484.0,
                    'projected_2030': 1089.0,
                    'cagr': '14.5%',
                    'growth_drivers': ['Video streaming growth', 'Social commerce', 'Creator economy', 'Mobile-first content'],
                    'key_trends': ['Short-form video', 'Live streaming', 'AR/VR content', 'Personalization'],
                    'challenges': ['Ad market volatility', 'Privacy regulations', 'Content moderation costs'],
                    'source': 'Mordor Intelligence 2024'
                },
                'electric vehicles': {
                    'market_size_2024': 562.0,
                    'projected_2030': 1579.0,
                    'cagr': '18.7%',
                    'growth_drivers': ['Climate regulations', 'Battery cost reduction', 'Charging infrastructure', 'Government incentives'],
                    'key_trends': ['Solid-state batteries', 'Autonomous features', 'Vehicle-to-grid', 'Platform sharing'],
                    'challenges': ['Raw material costs', 'Competition intensity', 'Profitability pressure'],
                    'source': 'Allied Market Research 2024'
                },
                'cloud computing': {
                    'market_size_2024': 679.0,
                    'projected_2030': 2432.0,
                    'cagr': '23.8%',
                    'growth_drivers': ['Hybrid cloud adoption', 'Edge computing', 'AI workloads', 'Enterprise migration'],
                    'key_trends': ['Multi-cloud strategy', 'Serverless computing', 'Kubernetes adoption', 'FinOps'],
                    'challenges': ['Data sovereignty', 'Vendor lock-in', 'Security compliance'],
                    'source': 'MarketsandMarkets 2024'
                },
                'ai_infrastructure': {
                    'market_size_2024': 147.0,  # Billion USD
                    'projected_2030': 732.0,
                    'cagr': '30.5%',
                    'growth_drivers': ['GenAI explosion', 'Enterprise AI adoption', 'LLM training demand', 'Edge AI deployment'],
                    'key_trends': ['GPU clusters', 'HBM memory adoption', 'AI-specific chips', 'Liquid cooling'],
                    'challenges': ['Power consumption', 'Chip supply constraints', 'Cooling requirements'],
                    'source': 'Grand View Research 2024 (AI Infrastructure)'
                },
                'telecommunications': {
                    'market_size_2024': 1850.0,
                    'projected_2030': 2560.0,
                    'cagr': '5.5%',
                    'growth_drivers': ['5G rollout', 'Network densification', 'Edge computing', 'Private networks'],
                    'key_trends': ['Open RAN', 'Network slicing', '6G research', 'Satellite integration'],
                    'challenges': ['Capex burden', 'Spectrum costs', 'Competition'],
                    'source': 'GSMA Intelligence 2024'
                },
                'biotechnology': {
                    'market_size_2024': 1537.0,
                    'projected_2030': 3440.0,
                    'cagr': '14.3%',
                    'growth_drivers': ['Gene therapy advances', 'Personalized medicine', 'CRISPR applications', 'Aging population'],
                    'key_trends': ['mRNA technology', 'CAR-T therapies', 'AI drug discovery', 'Biosimilars'],
                    'challenges': ['Regulatory approval times', 'R&D costs', 'Patent cliffs'],
                    'source': 'Grand View Research 2024'
                },
                'e-commerce': {
                    'market_size_2024': 6310.0,
                    'projected_2030': 16215.0,
                    'cagr': '17.0%',
                    'growth_drivers': ['Mobile commerce', 'Social commerce', 'Cross-border trade', 'Quick commerce'],
                    'key_trends': ['Live shopping', 'AR try-on', 'Sustainable packaging', 'Same-day delivery'],
                    'challenges': ['Last-mile costs', 'Returns management', 'Competition'],
                    'source': 'Statista Market Insights 2024'
                }
            }
            
            # Try to match industry
            matched_data = None
            primary_industry_name = None
            for key, data in industry_data.items():
                if key in industry or key in sector:
                    matched_data = data
                    primary_industry_name = key
                    break
            
            # Detect secondary industries from news content
            secondary_industries = []
            best_growth_rate = None
            best_industry_data = matched_data
            best_industry_name = primary_industry_name
            
            if news_items and matched_data:
                detected_industries = self.detect_secondary_industries(news_items)
                
                for sec_ind in detected_industries:
                    if sec_ind in industry_data and sec_ind != primary_industry_name:
                        sec_data = industry_data[sec_ind]
                        secondary_industries.append({
                            'name': sec_ind.replace('_', ' ').title(),
                            'cagr': sec_data['cagr'],
                            'detected_from': 'news_analysis'
                        })
                        
                        # Use the highest growth industry for scoring
                        sec_cagr = float(sec_data['cagr'].replace('%', ''))
                        primary_cagr = float(matched_data['cagr'].replace('%', ''))
                        
                        if best_growth_rate is None or sec_cagr > best_growth_rate:
                            best_growth_rate = sec_cagr
                            best_industry_data = sec_data
                            best_industry_name = sec_ind
            
            if not matched_data:
                return {'has_data': False, 'message': f'No market data available for industry: {industry}'}
            
            # Use the best (highest growth) industry data for the response
            response = {
                'has_data': True,
                'industry_name': (primary_industry_name or industry).replace('_', ' ').title(),
                'market_size_2024': best_industry_data['market_size_2024'],
                'projected_2030': best_industry_data['projected_2030'],
                'cagr': best_industry_data['cagr'],
                'growth_multiplier': round(best_industry_data['projected_2030'] / best_industry_data['market_size_2024'], 1),
                'growth_drivers': best_industry_data['growth_drivers'],
                'key_trends': best_industry_data['key_trends'],
                'challenges': best_industry_data['challenges'],
                'data_source': best_industry_data['source']
            }
            
            # Add secondary industry information if detected
            if secondary_industries:
                response['secondary_industries'] = secondary_industries
                response['multi_industry_note'] = f"Also serves {len(secondary_industries)} additional high-growth industry/industries"
                if best_industry_name != primary_industry_name:
                    response['primary_display_note'] = f"Using {best_industry_name.replace('_', ' ').title()} growth data (highest CAGR)"
            
            return response
            
        except Exception as e:
            print(f"Industry insights error: {e}")
            return {'has_data': False, 'error': str(e)}

def search_reddit_mentions(ticker, company_name=None, limit=15):
    """
    Search Reddit for mentions of a stock ticker with quality filtering.
    Focuses on reliable subreddits and filters out spam/pump posts
    Also checks if ticker or company name appears in the post
    """
    try:
        reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'StockAnalyzer/1.0')
        
        if not reddit_client_id or not reddit_client_secret:
            return {'info': 'Reddit API credentials not configured. Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env file'}
        
        import praw
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # Prioritized subreddits by reliability
        # Tier 1: High quality fundamental analysis
        tier1_subs = ['investing', 'stocks', 'StockMarket', 'ValueInvesting', 'SecurityAnalysis']
        # Tier 2: Active but mixed quality
        tier2_subs = ['wallstreetbets', 'options', 'Daytrading']
        
        mentions = []
        seen_titles = set()  # Avoid duplicates
        
        # Search tier 1 first (more reliable)
        for subreddit_name in tier1_subs:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                for submission in subreddit.search(f'${ticker} OR {ticker}', limit=5, sort='relevance', time_filter='month'):
                    
                    # Quality filters
                    if submission.title in seen_titles:
                        continue
                    
                    # Filter out low-quality posts
                    if submission.score < 5:  # Minimum upvotes
                        continue
                    
                    # Filter spam keywords
                    spam_keywords = ['moon', '🚀', 'pump', 'to the moon', 'yolo', 'meme']
                    title_lower = submission.title.lower()
                    spam_count = sum(1 for keyword in spam_keywords if keyword in title_lower)
                    
                    # Check if ticker or company name is actually mentioned
                    title_lower = submission.title.lower()
                    text_lower = (submission.selftext or '').lower()
                    ticker_lower = ticker.lower()
                    
                    # Check for ticker variants
                    ticker_variants = [f'${ticker_lower}', ticker_lower, f' {ticker_lower} ', f' {ticker_lower}.']
                    has_ticker = any(variant in title_lower or variant in text_lower for variant in ticker_variants)
                    
                    # Also check for company name if provided
                    has_company_name = False
                    if company_name:
                        # Split company name into parts to match partial names (e.g., "Micron" from "Micron Technology")
                        company_parts = company_name.lower().split()
                        # Check if any significant part of the company name appears (skip common words like "inc", "corp")
                        significant_parts = [part for part in company_parts if part not in ['inc', 'inc.', 'corp', 'corp.', 'ltd', 'ltd.', 'company', 'technologies', 'technology']]
                        if significant_parts:
                            has_company_name = any(part in title_lower or part in text_lower for part in significant_parts if len(part) > 3)
                    
                    if not (has_ticker or has_company_name):
                        continue
                    
                    # Calculate quality score
                    quality_score = submission.score + (submission.num_comments * 2)
                    if spam_count >= 2:
                        quality_score *= 0.5  # Penalize spam
                    
                    seen_titles.add(submission.title)
                    mentions.append({
                        'title': submission.title,
                        'score': submission.score,
                        'url': f"https://reddit.com{submission.permalink}",
                        'created': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M'),
                        'num_comments': submission.num_comments,
                        'subreddit': subreddit_name,
                        'quality_score': quality_score,
                        'tier': 1,
                        'is_spam': spam_count >= 2
                    })
            except Exception as e:
                print(f"Error searching r/{subreddit_name}: {e}")
                continue
        
        # Search tier 2 if we need more results
        if len(mentions) < 5:
            for subreddit_name in tier2_subs:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(f'${ticker} OR {ticker}', limit=5, sort='hot', time_filter='week'):
                        
                        if submission.title in seen_titles:
                            continue
                        
                        if submission.score < 10:  # Higher threshold for tier 2
                            continue
                        
                        spam_keywords = ['moon', '🚀', 'pump', 'to the moon', 'yolo', 'meme']
                        title_lower = submission.title.lower()
                        text_lower = (submission.selftext or '').lower()
                        spam_count = sum(1 for keyword in spam_keywords if keyword in title_lower)
                        
                        # Check for ticker or company name
                        ticker_lower = ticker.lower()
                        ticker_variants = [f'${ticker_lower}', ticker_lower, f' {ticker_lower} ', f' {ticker_lower}.']
                        has_ticker = any(variant in title_lower or variant in text_lower for variant in ticker_variants)
                        
                        has_company_name = False
                        if company_name:
                            company_parts = company_name.lower().split()
                            significant_parts = [part for part in company_parts if part not in ['inc', 'inc.', 'corp', 'corp.', 'ltd', 'ltd.', 'company', 'technologies', 'technology']]
                            if significant_parts:
                                has_company_name = any(part in title_lower or part in text_lower for part in significant_parts if len(part) > 3)
                        
                        if not (has_ticker or has_company_name):
                            continue
                        
                        quality_score = submission.score + (submission.num_comments * 2)
                        if spam_count >= 2:
                            quality_score *= 0.3  # Heavy penalty for tier 2 spam
                        
                        seen_titles.add(submission.title)
                        mentions.append({
                            'title': submission.title,
                            'score': submission.score,
                            'url': f"https://reddit.com{submission.permalink}",
                            'created': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M'),
                            'num_comments': submission.num_comments,
                            'subreddit': subreddit_name,
                            'quality_score': quality_score,
                            'tier': 2,
                            'is_spam': spam_count >= 2
                        })
                except Exception as e:
                    print(f"Error searching r/{subreddit_name}: {e}")
                    continue
        
        # Sort by quality score and return top results
        mentions = sorted(mentions, key=lambda x: x['quality_score'], reverse=True)[:limit]
        
        # Filter out spam from final results (unless it's all we have)
        clean_mentions = [m for m in mentions if not m['is_spam']]
        if len(clean_mentions) >= 3:
            mentions = clean_mentions
        
        return mentions if mentions else {'info': f'No quality Reddit discussions found for ${ticker} in the past month'}
        
    except Exception as e:
        print(f"Reddit search error: {e}")
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    ticker = data.get('ticker', '').strip().upper()
    
    if not ticker:
        return jsonify({'error': 'Please provide a stock ticker'}), 400
    
    # Check if user entered a company name instead of ticker
    # Allow dots (.) for international tickers (e.g., AUS.F for Frankfurt)
    # Allow hyphens (-) for some tickers (e.g., BRK-B)
    if ' ' in ticker or len(ticker) > 10:
        return jsonify({
            'error': f'Please enter a stock ticker symbol (e.g., "MU", "AUS.F") instead of company name. '
                    f'Ticker symbols are usually 1-10 characters.'
        }), 400
    
    # Check cache first to reduce API calls
    current_time = time.time()
    if ticker in request_cache:
        cached_data = request_cache[ticker]
        cache_age = current_time - cached_data['timestamp']
        if cache_age < CACHE_DURATION:
            print(f"Returning cached data for {ticker} (age: {int(cache_age)}s)")
            return jsonify(cached_data['data'])
    
    try:
        analyzer = StockAnalyzer(ticker)
        
        # Gather all data
        basic_info = analyzer.get_basic_info()
        if 'error' in basic_info:
            return jsonify({
                'error': f'Could not find data for ticker "{ticker}". Please verify the ticker symbol is correct. '
                        f'Example: Use "MU" for Micron Technology, "AAPL" for Apple, etc.'
            }), 400
        
        financial_metrics = analyzer.get_financial_metrics()
        industry_comparison = analyzer.get_industry_comparison()
        volume_analysis = analyzer.get_volume_analysis(period='1mo')
        price_history = analyzer.get_price_history(period='1y')  # Default to 1 year
        analyst_chart_data = analyzer.get_analyst_estimates_chart_data()
        news = analyzer.get_news()
        industry_trend = analyzer.analyze_industry_trend()
        
        # Get industry insights with news analysis for multi-industry detection
        industry_insights = analyzer.get_industry_insights(news_items=news)
        
        # Pass industry insights to business outlook for comprehensive scoring
        business_outlook = analyzer.get_business_outlook(days=90, industry_insights=industry_insights)
        
        # Get company name for Reddit filtering
        company_name = basic_info.get('name', '')
        reddit_mentions = search_reddit_mentions(ticker, company_name)
        
        profitability = analyzer.get_profitability_status()
        
        response = {
            'basic_info': basic_info,
            'financial_metrics': financial_metrics,
            'industry_comparison': industry_comparison,
            'volume_analysis': volume_analysis,
            'price_history': price_history,
            'analyst_chart_data': analyst_chart_data,
            'news': news,
            'industry_trend': industry_trend,
            'business_outlook': business_outlook,
            'reddit_mentions': reddit_mentions,
            'profitability': profitability,
            'industry_insights': industry_insights,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Clean NaN values before returning
        response = clean_nan(response)
        
        # Cache the successful response
        request_cache[ticker] = {
            'data': response,
            'timestamp': current_time
        }
        print(f"Cached response for {ticker}")
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chart/<ticker>')
def get_chart(ticker):
    """Generate interactive price chart"""
    try:
        period = request.args.get('period', '6mo')
        analyzer = StockAnalyzer(ticker)
        price_data = analyzer.get_price_history(period)
        
        if 'error' in price_data:
            return jsonify({'error': price_data['error']}), 400
        
        # Calculate percentage changes from first value for hover display
        if len(price_data['prices']) > 0:
            first_price = price_data['prices'][0]
            price_data['percent_changes'] = [
                round(((price - first_price) / first_price * 100), 2) if first_price > 0 else 0
                for price in price_data['prices']
            ]
        
        # Clean NaN values before returning
        price_data = clean_nan(price_data)
        return jsonify(price_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use debug mode only in development
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.getenv('PORT', 5001)))

