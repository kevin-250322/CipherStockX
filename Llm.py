import google.generativeai as genai
import os
from typing import List
from pydantic import BaseModel
import json
import requests
from bs4 import BeautifulSoup
import yfinance as yf

class StockInfo(BaseModel):
    ticker: str
    earnings_growth: str
    revenue_growth: str
    ebitda_margin: str
    gross_margin: str
    fifty_two_week_change: str
    summary: str

class StockComparisonSummary(BaseModel):
    stock_comparison: List[StockInfo]
    overall_summary: str

class llm:
    def __init__(self):
        genai.configure(api_key=os.getenv("API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash-8b")
        self.prompt = """
                    Please provide a detailed stock comparison summary in JSON format. Compare each stock based on these metrics:
                    1. Earnings growth
                    2. Revenue growth
                    3. EBITDA margins
                    4. Gross margins
                    5. 52-week change

                    For each stock, create an individual summary in JSON, followed by a brief paragraph explaining the comparison results in a human-readable format. Ensure the output is well-structured in JSON.

                    Input JSON for stocks:
                    {stock_info_json}

                    Output format:
                    {{
                        "stock_comparison": [
                            {{
                                "ticker": "AMT",
                                "earnings_growth": "88.20%",
                                "revenue_growth": "4.60%",
                                "ebitda_margin": "71.45%",
                                "gross_margin": "52.71%",
                                "52_week_change": "29.41%",
                                "summary": "Stock AMT has the highest earnings growth (88.20%) and gross margins (71.45%), along with a strong
                                52-week change (29.41%), but its revenue growth (4.60%) is lower compared to EQIX and DLR."
                            }},
                            ...
                        ],
                        "overall_summary": "In Summary: This comparison highlights a variety of performance metrics among the stocks. AMT stands out for its significant earnings growth and high margins, while EQIX offers a balance between growth and decent margins. DLR provides strong margins albeit with declining growth, and HPE struggles with both low growth and margins. NTTYY exhibits the lowest performance across multiple metrics. Investors may prioritize different features based on their risk appetite and investment objectives."
                    }}

                    later on i shld be able to use like this json like this to print 
                    '
                    Stock Comparison Summary
                    • Stock AMT has the highest earnings growth (88.20%) and gross margins (71.45%), along with a strong
                    52-week change (29.41%), but its revenue growth (4.60%) is lower compared to EQIX and DLR.
                    • Stock EQIX has impressive earnings growth (43.00%) and consistent revenue growth (6.90%), with solid EBITDA margins (37.71%), but its gross margins are lower than AMT and DLR.
                    • Stock DLR has the highest gross margins (52.71%) and EBITDA margins (43.63%), but it faces A significant declines in both earnings growth (-41.10%) and revenue growth (-4.10%), despite a reasonably high 52-week change (15.24%).
                    • Stock HPE displays declining earnings growth (-25.20%) and modest revenue growth (3.30%), with the lowest gross margins (34.97%) and EBITDA margins (17.28%), alongside a moderate 52-week change (7.38%).
                    • Stock NTTYY shows declining earnings growth (-26.00%) and relatively lower gross margins (29.14%)
                    compared to its peers, with a negative 52-week change (-6.15%).
                    In Summary: This comparison highlights a variety of performance metrics among the stocks. AMT stands out for its significant earnings growth and high margins, while EQIX offers a balance between growth and decent margins. DLR provides strong margins albeit with declining growth, and HPE struggles with both low growth and margins. NTTYY exhibits the lowest performance across multiple metrics. Investors may prioritize different features based on their risk appetite and investment objectives.'
                    MAKE SURE TO USE % METRICS IN SUMMARY OF EACH TICKER and ONLY OUTPUT JSON AND NO PREAMBEL OR POST COMMNETS
                    """
        self.prompt2="""
                    perform sentiment analysis for each ticker and give analysis(write in the tone of an analysit and mentioning number and 
                    comprasion data if possible) and sentiment score(0-100) output in json format here is the input json {sentiment_analysis}

                    Example output format:
                    {{
                    "AMT": {{
                        "analysis": str,
                        "sentiment_score": 95
                    }},
                    ........
                    }}
                    """
        self.prompt3="""Refine the input '{keyword}' to add context for accurately identifying companies in this sector.
                    Return a precise, single sentence describing the companies or sector relevant to '{keyword}' in a way that enhances search specificity.
                    For example, 'data center builders' becomes 'companies specializing in designing, constructing, and supplying technology for data storage and processing centers.'
                    Note: NO Preambel"""


    def find_braces_positions(self,s: str):
        first_brace = s.find('{')  # Find the position of the first '{'
        last_brace = s.rfind('}')  # Find the position of the last '}'
        return first_brace, last_brace
    
    def basiccomapre(self,stock_info_json):
        # Generating structured output using Gemini's JSON mode
        response = self.model.generate_content(self.prompt.format(stock_info_json=stock_info_json))
        Start,End=self.find_braces_positions(response.text)
        return json.loads(response.text[Start:End+1])


    def UrlTextScrape(self,url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            page_text = soup.get_text(separator=' ', strip=True)
            return page_text

        except requests.RequestException as e:
            return
    


    def sentiment_data(self,tickers):
        sentiments={}
        for tick in tickers:
            stock = yf.Ticker(tick)
            sentiments[tick] = []
            for i in range(0,len(stock.news)):
                if '/m/' not in stock.news[i]['link']:
                    text = self.UrlTextScrape(stock.news[i]['link'])
                    sentiments[tick].append(
                {
                    'date': stock.news[i]['providerPublishTime'],
                    'title': stock.news[i]['title'],
                    'text': text,
                    'liink': stock.news[i]['link'],
                    'relatedTickers': stock.news[i]['relatedTickers']

                }
            )
        return json.dumps(sentiments)#sentiment_analysis
    
    def sentiment_analysis_tick(self,sentiment_analysis):
        response2 = self.model.generate_content(self.prompt2.format(sentiment_analysis=sentiment_analysis))
        Start1,End1=self.find_braces_positions(response2.text)
        print(response2.text[Start1:End1+1])
        return json.loads(response2.text[Start1:End1+1])#sentiment_analysis_dict
    
    def eloberateprompt(self,keywords):
        response3=self.model.generate_content(self.prompt3.format(keyword=keywords))
        return response3.text


