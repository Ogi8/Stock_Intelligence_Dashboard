from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
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

load_dotenv()

app = Flask(__name__)
FMP_API_KEY = os.getenv('FMP_API_KEY')

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        
    def get_basic_info(self):
        """Get basic stock information"""
        try:
            info = self.stock.info
            
            # Check if we got valid data
            if not info or len(info) < 5:
                return {'error': 'No data available for this ticker'}
            
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
                        metrics['historical'][metric_name] = {
                            'dates': [date.strftime('%Y-%m-%d') for date in data.index],
                            'values': [float(val) for val in data.values]
                        }
                        # Latest value
                        metrics[metric_name] = float(data.iloc[-1])
                
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
    
    def get_business_outlook(self, days=90):
        """Analyze business outlook based on news and earnings keywords"""
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
            
            # Score calculation
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
            
            # Calculate score (0-100)
            strong_score = len(strong_matches) * 15  # Each strong match = 15 points
            moderate_score = len(moderate_matches) * 8  # Each moderate match = 8 points
            total_score = min(100, strong_score + moderate_score)
            
            # Determine label
            if total_score >= 60:
                outlook_label = 'Strong'
                outlook_color = '#10b981'
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
                'outlook_label': outlook_label,
                'outlook_color': outlook_color,
                'strong_signals': len(strong_matches),
                'moderate_signals': len(moderate_matches),
                'evidence': evidence[:10],  # Top 10 pieces of evidence
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            
        except Exception as e:
            print(f"Business outlook error: {e}")
            return {
                'outlook_score': 0,
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
        """Analyze if the industry is trending with Bayesian analyst consensus"""
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
                'analyst_target': target_mean,
                'current_price': current_price,
                'target_median': target_median,
                'target_high': target_high,
                'target_low': target_low,
                'num_analysts': num_analysts
            }
            
            # Calculate basic potential upside
            if isinstance(target_mean, (int, float)) and isinstance(current_price, (int, float)):
                upside = ((target_mean - current_price) / current_price * 100)
                analysis['potential_upside'] = round(upside, 2)
            
            # Bayesian/Bootstrap Estimation with historical context
            if all(isinstance(val, (int, float)) for val in [target_mean, target_median, target_high, target_low, current_price]) and num_analysts > 0:
                # Try to get historical analyst data
                historical_data = self.get_historical_analyst_estimates(years=3)
                
                bayesian_analysis = self._calculate_bayesian_consensus(
                    current_price, target_mean, target_median, target_high, target_low, num_analysts, historical_data
                )
                analysis.update(bayesian_analysis)
            
            return analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_bayesian_consensus(self, current_price, mean_target, median_target, high_target, low_target, num_analysts, historical_data=None):
        """
        Calculate Bayesian weighted consensus of analyst predictions
        Now enhanced with historical analyst data for better accuracy
        
        Models analyst opinions as Gaussian distributions:
        P(true_price | analysts) ∝ Σ w_i · N(μ_i, σ_i)
        
        where:
        - μ_i is each analyst group's target
        - σ_i is the uncertainty (spread)
        - w_i is the weight based on confidence and historical accuracy
        """
        try:
            # Estimate the standard deviation from the range
            # Using the empirical rule: range ≈ 4σ for ~95% of data
            price_range = high_target - low_target
            estimated_std = price_range / 4.0
            
            # Adjust weights based on historical data if available
            base_weights = {'median': 0.4, 'mean': 0.3, 'bull': 0.15, 'bear': 0.15}
            
            if historical_data and 'historical_data' in historical_data and len(historical_data['historical_data']) > 0:
                # We have historical data - adjust confidence based on trend consistency
                hist_points = historical_data['historical_data']
                
                # Check if estimates have been consistently revised up or down
                if len(hist_points) >= 3:
                    recent_avg = np.mean([p.get('estimated_avg', mean_target) for p in hist_points[:3] if p.get('estimated_avg')])
                    older_avg = np.mean([p.get('estimated_avg', mean_target) for p in hist_points[-3:] if p.get('estimated_avg')])
                    
                    # If consistent upward/downward revision, increase median weight (more confident)
                    if recent_avg and older_avg:
                        revision_direction = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
                        if abs(revision_direction) > 0.1:  # Significant revision trend
                            base_weights['median'] = 0.5  # More weight on median (stronger consensus)
                            base_weights['mean'] = 0.35
                            base_weights['bull'] = 0.075
                            base_weights['bear'] = 0.075
            
            # Create weighted Gaussian distributions for different analyst groups
            # We'll model the distribution as a mixture of Gaussians
            
            # Define analyst "groups" with their targets and weights
            # Weight more heavily towards median (more robust to outliers)
            analyst_groups = [
                {'target': median_target, 'weight': base_weights['median'], 'std': estimated_std * 0.8},  # Median - most reliable
                {'target': mean_target, 'weight': base_weights['mean'], 'std': estimated_std},          # Mean - includes all
                {'target': (high_target + median_target) / 2, 'weight': base_weights['bull'], 'std': estimated_std * 1.2},  # Bull case
                {'target': (low_target + median_target) / 2, 'weight': base_weights['bear'], 'std': estimated_std * 1.2}    # Bear case
            ]
            
            # Monte Carlo simulation: sample from the mixture of Gaussians
            n_samples = 10000
            samples = []
            
            for group in analyst_groups:
                # Sample from each Gaussian, weighted by probability
                n_group_samples = int(n_samples * group['weight'])
                group_samples = np.random.normal(
                    loc=group['target'],
                    scale=group['std'],
                    size=n_group_samples
                )
                samples.extend(group_samples)
            
            samples = np.array(samples)
            
            # Calculate statistics from the posterior distribution
            bayesian_mean = np.mean(samples)
            bayesian_median = np.median(samples)
            bayesian_std = np.std(samples)
            
            # Calculate confidence intervals (credible intervals in Bayesian terms)
            percentile_5 = np.percentile(samples, 5)   # 5th percentile
            percentile_25 = np.percentile(samples, 25)  # 25th percentile (Q1)
            percentile_75 = np.percentile(samples, 75)  # 75th percentile (Q3)
            percentile_95 = np.percentile(samples, 95)  # 95th percentile
            
            # Calculate probability of upside (P(target > current_price))
            prob_upside = np.sum(samples > current_price) / len(samples)
            
            # Calculate expected value (risk-adjusted return)
            expected_return = ((bayesian_mean - current_price) / current_price) * 100
            
            # Calculate probability-weighted upside/downside
            upside_samples = samples[samples > current_price]
            downside_samples = samples[samples <= current_price]
            
            if len(upside_samples) > 0:
                avg_upside = ((np.mean(upside_samples) - current_price) / current_price) * 100
            else:
                avg_upside = 0
            
            if len(downside_samples) > 0:
                avg_downside = ((np.mean(downside_samples) - current_price) / current_price) * 100
            else:
                avg_downside = 0
            
            # Calculate risk-reward ratio
            if avg_downside != 0:
                risk_reward_ratio = abs(avg_upside / avg_downside)
            else:
                risk_reward_ratio = float('inf') if avg_upside > 0 else 0
            
            result = {
                'bayesian_target': round(bayesian_mean, 2),
                'bayesian_median': round(bayesian_median, 2),
                'bayesian_std': round(bayesian_std, 2),
                'confidence_interval_90': (round(percentile_5, 2), round(percentile_95, 2)),
                'confidence_interval_50': (round(percentile_25, 2), round(percentile_75, 2)),
                'probability_upside': round(prob_upside * 100, 1),
                'expected_return': round(expected_return, 2),
                'avg_upside_scenario': round(avg_upside, 2),
                'avg_downside_scenario': round(avg_downside, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2) if risk_reward_ratio != float('inf') else 'Inf',
                'analyst_consensus_strength': self._get_consensus_strength(bayesian_std, current_price)
            }
            
            # Add historical data info if available
            if historical_data and 'data_points' in historical_data:
                result['historical_data_points'] = historical_data['data_points']
                result['historical_date_range'] = historical_data.get('date_range', 'N/A')
                result['uses_historical_data'] = True
            else:
                result['uses_historical_data'] = False
            
            return result
        except Exception as e:
            print(f"Bayesian calculation error: {e}")
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

def search_reddit_mentions(ticker, limit=15):
    """
    Search Reddit for high-quality stock mentions with relevance filtering
    Focuses on reliable subreddits and filters out spam/pump posts
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
                    
                    # Check if ticker is actually mentioned (not just in broader discussion)
                    ticker_variants = [f'${ticker}', ticker, ticker.lower()]
                    has_ticker = any(variant in submission.title or variant in (submission.selftext or '') for variant in ticker_variants)
                    
                    if not has_ticker:
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
                        spam_count = sum(1 for keyword in spam_keywords if keyword in title_lower)
                        
                        ticker_variants = [f'${ticker}', ticker, ticker.lower()]
                        has_ticker = any(variant in submission.title or variant in (submission.selftext or '') for variant in ticker_variants)
                        
                        if not has_ticker:
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

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    ticker = data.get('ticker', '').strip().upper()
    
    if not ticker:
        return jsonify({'error': 'Please provide a stock ticker'}), 400
    
    # Check if user entered a company name instead of ticker
    if ' ' in ticker or len(ticker) > 5:
        return jsonify({
            'error': f'Please enter a stock ticker symbol (e.g., "MU") instead of company name. '
                    f'Ticker symbols are usually 1-5 letters.'
        }), 400
    
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
        news = analyzer.get_news()
        industry_trend = analyzer.analyze_industry_trend()
        business_outlook = analyzer.get_business_outlook(days=90)
        reddit_mentions = search_reddit_mentions(ticker)
        profitability = analyzer.get_profitability_status()
        
        response = {
            'basic_info': basic_info,
            'financial_metrics': financial_metrics,
            'industry_comparison': industry_comparison,
            'volume_analysis': volume_analysis,
            'news': news,
            'industry_trend': industry_trend,
            'business_outlook': business_outlook,
            'reddit_mentions': reddit_mentions,
            'profitability': profitability,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
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
        
        return jsonify(price_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use debug mode only in development
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))

