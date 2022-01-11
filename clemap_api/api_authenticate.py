import requests
from requests.structures import CaseInsensitiveDict
import json

class clemap_api():

    def __init__(self, email, password):
        self.request_url = 'https://docs.cloud.clemap.com/authentication'
        self.email = email
        self.password = password
    
    def authenticate(self):
        url = 'https://docs.cloud.clemap.com/authentication'
        headers = CaseInsensitiveDict()
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json"
        
        data = {"strategy": "local", 
                "email": self.email, 
                "password": self.password}

        data_json = json.dumps(data)

        resp = requests.post(url, headers=headers, data=data_json)
        access_token = resp.content
        access_token = json.loads(access_token)
        self.access_token = list(access_token.values())[0]
        
        return self.access_token
    
    def power_data(self, sensor_id, granularity, start_day, end_day, start_time, end_time):
        request_url = 'https://docs.cloud.clemap.com/pdata?sensor_id='+sensor_id+'&granularity='+granularity+'&start='+start_day+granularity.upper()+start_time+'Z&end='+end_day+granularity.upper()+end_time+'Z'
        headers = CaseInsensitiveDict()
        headers["Accept"] = "application/json"
        headers['Authorization'] = self.access_token

        power_data = requests.get(request_url, headers=headers)

        return power_data.content