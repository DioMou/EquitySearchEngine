# FinancialSearchEngine

Purpose:
In this project, a search engine solution that serves to direct investors to a compressed list of companies for deep investigation is provided. This search engine solution is composed of two sections. First, the search engine is going to find out relevant companies that match the investors’ mind and return back necessary financial balance sheet information for the company. For example, the search engine is going to report back agricultural companies with certain financial data if the investor is planning to investigate companies in agricultural field. Second, the search engine is going to rank the companies based on the investment potentials for those selected companies. In the end, it will return a list of companies ranked from top to bottom that inform investors the compressed list of companies to investigate based on the financial data. The compressed company list is currently designed to contain 20 companies.


Here are the list of libraries needed:
pandas
numpy
pytickersymbols
urllib3
beautifulsoup4
fastrank
python-terrier


To run the model, simpy use the following command
python main.py "your search query"

Example input and output(list of string, the string represents the yahoo company stickers):
python main.py "cannabinoid molecules drug"

['INM', 'OPNT', 'NUVB', 'SCPS', 'SDGR', 'CRIS', 'ANEB', 'VYNT', 'RAPT', 'CLVRW', 'ENVB', 'CRBP', 'PMD', 'SPRC', 'EXAI', 'LEXX', 'CLVR', 'LZAGY', 'LZAGF', '0QNO.L']

