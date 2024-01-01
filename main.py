import requests

def prompt_fbn_norm(session_id, prompt):
    url = 'https://www.fbn.com/api/notifications/norm/chat/prompt'

    headers = {
        'authority': 'www.fbn.com',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/json',
        'dnt': '1',
        'origin': 'https://www.fbn.com',
        'referer': 'https://www.fbn.com/norm',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-csrf-token': '3d2fa262a0abae7bc4cafee703dc1c7ebf4a83536b5b17dbedafe2c31849d7f5',
        'x-datadog-trace-id': '926791915438938916',
        'x-js-build-tag': 'build-642bd252496f7104-3944001',
        'x-requested-with': 'XMLHttpRequest',
        # Add your cookies here
        'cookie': '_ga=GA1.1.1066855511.1703907093; _mkto_trk=id:197-HFR-752&token:_mch-fbn.com-1703907093092-42493; _gcl_au=1.1.779175757.1703907093; ajs_anonymous_id=6f729cff-ba40-4b95-8652-334aceffceaf; _fbp=fb.1.1703907093912.176531731; landingPageURLRaw=aHR0cHM6Ly93d3cuZmJuLmNvbS9ub3Jt; landingPageURL=aHR0cHM6Ly93d3cuZmJuLmNvbS9ub3Jt; referrerPageURLRaw=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8=; referrerPageURL=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8=; formURLRaw=aHR0cHM6Ly93d3cuZmJuLmNvbS9zaWdudXA=; formURL=aHR0cHM6Ly93d3cuZmJuLmNvbS9zaWdudXA=; ajs_user_id=165467; user_logged_in=1; fbnAuth=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJjb29raWUiOnRydWUsImV4cCI6MTcxODQyMjQzMywianRpIjoiOGRkZDcxNWVhZTAyNjJjYzY5YmEwNDQ5NGI3ODBlYjlkZDE4MTE1MWQ1MjExNDZjMjAzNWIyMmI1MzJhMzIzMSIsImtleSI6ImU1ZDkyMzk4OTVmYjM1MDBjNzdkNmRkMjAxNWU3MmQzYTgzNGM2NTZmNzU0ZTZkNzgwZjNhZmVhZDVhOWI5ODQiLCJzdWIiOjE2NTQ2NywidHlwIjoibG9naW4iLCJ2ZXIiOjF9.giRUTmvRPNiAPDvV1mk3uNmZcYtn94-cEcUftAGZQFb7XT9EqwPnl4coJEpT0aL0Skqfozimy9Uhcqgsj2-WQbBDI_LkNDbxGL8cbhZDFvgVQfEH_H5iH6D0OPjvc38VTGs_1gx-23dO5wKpVAcehMwOyySkBjYwq-S84XhEUhlb_zf4oTMqx4y0qcBRSMvfVqyyxeeINv27fgznFWFkEO--ilO1NH979xvRrpauc4gTPjLnca26dc244TIfp7YerkD1uh5PCQinE80HseSG8gttxoVMec2rcdyYMASHGlHwX-PmAmfstGZQtsWSWLq0Gxi-kGHTdlicihcfBoUOqQ; _clck=i1rp5i%7C2%7Cfi0%7C0%7C1459; analytics_session_id=1704008492867; analytics_session_id.last_access=1704009340247; _uetsid=ed9cd400a6c311ee852bfb47b0eaf23e; _uetvid=ed9ce600a6c311eebcbead0030aa6b92; _ga_E36VZHGD7X=GS1.1.1704008492.3.1.1704009340.58.0.0; _clsk=9ib4w4%7C1704009340495%7C2%7C1%7Ca.clarity.ms%2Fcollect; last_activity_timestamp=1704009340795'  # truncated for brevity
    }

    data = {
        'session_id': session_id,
        #'postal_code': postal_code,
        #'country_code': country_code,
        'prompt': prompt
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example usage
#response = prompt_fbn_norm("8e9437f8-8d15-4cb9-7da1-b376b3fb0a8f", "hello")
with open('quizlet.txt', 'r') as file:
    lines = file.readlines()

questions = []
question = {}
answers = {}
answer_key = None
for line in lines:
    line = line.strip()
    if line == '':
        question['answers'] = answers
        questions.append(question)
        question = {}
        answers = {}
    elif line in ['a.', 'b.', 'c.', 'd.']:
        answer_key = line[0]
    elif len(line) == 1:
        question['answer'] = line
    elif 'question' not in question:
        question['question'] = line
    else:
        if answer_key:
            answers[answer_key] = line

for question in questions:
    print(question)

