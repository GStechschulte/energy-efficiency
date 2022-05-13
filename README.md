# Energy-Efficiency-Thesis

## Reproduce the results

1.) Clone the repo

2.) Build image

3.) Run the container and model(s)

`docker run -i thesis-model`

You will be asked to enter a machine and the time aggregation you would like to analyze. Available time sampling is `10` and `30` minutes. The following devices have energy baseline models ready for inference:

- Entsorgung
- Hauptluftung
- Gesamtmessung
- UV EG
- UV OG
- EG