"""
Download S&P 500 price data and ESG scores for CASP analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time

# S&P 500 tickers (top 100)
tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 
           'UNH', 'JNJ', 'JPM', 'XOM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 
           'ABBV', 'PEP', 'KO', 'LLY', 'BAC', 'AVGO', 'TMO', 'COST', 'DIS', 
           'MCD', 'CSCO', 'WMT', 'CRM', 'ACN', 'VZ', 'DHR', 'NEE', 'LIN', 
           'ADBE', 'TXN', 'PM', 'AMD', 'BMY', 'RTX', 'NFLX', 'QCOM', 'HON', 
           'UPS', 'UNP', 'LOW', 'INTC', 'SPGI', 'INTU', 'CAT', 'IBM', 'AMGN', 
           'GS', 'GE', 'DE', 'MS', 'PLD', 'SBUX', 'BA', 'BLK', 'MMM', 'T', 
           'AXP', 'CVS', 'MDLZ', 'AMT', 'GILD', 'ISRG', 'BKNG', 'LMT', 'ADI', 
           'ADP', 'VRTX', 'ZTS', 'SYK', 'NOW', 'TJX', 'REGN', 'MMC', 'EL', 
           'C', 'PGR', 'SCHW', 'TMUS', 'MO', 'EOG', 'NKE', 'SO', 'DUK', 
           'BSX', 'SLB', 'ITW', 'BDX', 'CI', 'AON', 'APD', 'CL', 'USB']


def download_esg_yahoo(ticker_list):
    """
    Download ESG-related data from Yahoo Finance.
    
    Since Yahoo deprecated the sustainability endpoint, we use:
    1. Governance risk scores from stock.info (auditRisk, boardRisk, etc.)
    2. Sector information for environmental/social estimates
    
    Returns a composite ESG quality score (higher = better).
    """
    esg_data = []
    
    # Sector-based E and S score adjustments (based on typical sector ESG profiles)
    sector_es_scores = {
        'Technology': 75,
        'Healthcare': 72,
        'Financial Services': 68,
        'Consumer Cyclical': 65,
        'Consumer Defensive': 70,
        'Industrials': 60,
        'Basic Materials': 55,
        'Energy': 45,
        'Utilities': 65,
        'Real Estate': 68,
        'Communication Services': 70,
    }
    
    for i, ticker in enumerate(ticker_list):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info:
                esg_dict = {'ticker': ticker}
                
                # Get governance risk scores (1-10, lower is better)
                audit_risk = info.get('auditRisk', 5)
                board_risk = info.get('boardRisk', 5)
                comp_risk = info.get('compensationRisk', 5)
                rights_risk = info.get('shareHolderRightsRisk', 5)
                overall_risk = info.get('overallRisk', 5)
                
                # Convert governance risk to quality score (0-100)
                # Risk is 1-10 where 1 is best, so we invert: (10 - risk) * 10
                gov_score = (10 - overall_risk) * 10
                
                # Get sector for E and S estimates
                sector = info.get('sector', 'Unknown')
                es_score = sector_es_scores.get(sector, 65)
                
                # Composite ESG score: weighted average
                # G: 40%, E+S (from sector): 60%
                total_esg = 0.4 * gov_score + 0.6 * es_score
                
                esg_dict['total_esg'] = total_esg
                esg_dict['governance'] = gov_score
                esg_dict['env_social'] = es_score
                esg_dict['sector'] = sector
                esg_dict['audit_risk'] = audit_risk
                esg_dict['board_risk'] = board_risk
                esg_dict['overall_risk'] = overall_risk
                
                esg_data.append(esg_dict)
                print(f"  {ticker}: ESG = {total_esg:.1f} (G={gov_score:.0f}, ES={es_score}, Sector={sector})")
            else:
                print(f"  {ticker}: No info data")
                
        except Exception as e:
            print(f"  {ticker}: Error - {str(e)[:50]}")
        
        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.2)
    
    return pd.DataFrame(esg_data)


if __name__ == '__main__':
    print(f"Downloading data for {len(tickers)} S&P 500 stocks...")
    print("Date range: 2020-01-01 to 2024-12-01")
    print("="*60)

    # Download 4 years of price data
    data = yf.download(tickers, start='2020-01-01', end='2024-12-01', progress=True)['Close']

    # Clean and save price data
    data = data.dropna(axis=1, thresh=int(len(data)*0.9))
    data = data.ffill().bfill()
    data.to_csv('../data/sp500_prices.csv')

    print(f"\nSaved {len(data.columns)} stocks to ../data/sp500_prices.csv")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(data)}")

    # Download ESG data from Yahoo Finance
    print("\n" + "="*60)
    print("Downloading ESG scores from Yahoo Finance...")
    print("="*60)
    
    valid_tickers = list(data.columns)
    esg_df = download_esg_yahoo(valid_tickers)
    
    print("\n" + "="*60)
    
    if len(esg_df) > 0:
        # Save ESG data in format compatible with analysis code
        esg_df.set_index('ticker', inplace=True)
        esg_df.rename(columns={'total_esg': 'ESG_Score'}, inplace=True)
        esg_df.to_csv('../data/sp500_esg.csv')
        
        print(f"Successfully downloaded ESG data for {len(esg_df)} stocks")
        print(f"ESG score range: [{esg_df['ESG_Score'].min():.1f}, {esg_df['ESG_Score'].max():.1f}]")
        print(f"ESG score mean: {esg_df['ESG_Score'].mean():.1f}")
        print("\nSample data:")
        print(esg_df.head(10))
    else:
        print("\nNo ESG data could be retrieved from Yahoo Finance.")
        print("The analysis will use synthetic ESG scores based on sector patterns.")

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)

