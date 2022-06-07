# Energy-Efficiency-Thesis

## Reproduce the results

![entsorgung_gp](./img/LocPer_RQ_entsorgung.png)
![entsorgung_spc](./img/entsorgung_test_SPC.png)
![entsorgung_spc](./img/entsorgung_deviation.png)

**Method 1**

1.) Clone the repo and `cd` into the root directory

2.) Build image

`docker build thesis-model .`

3.) Run the container and model(s)

`docker run -i thesis-model`

**Method 2**

1.) Pull the image from Docker Hub (contact author for access to private Docker Hub )

`docker pull [OPTIONS] NAME[:TAG|@DIGEST]`


### Available Data

You will be asked to enter a machine and the time aggregation you would like to analyze. Available time sampling is `10` and `30` minutes. The following devices have energy baseline models ready for inference:

- Entsorgung
- Hauptluftung
- Gesamtmessung
- UV EG
- UV OG
- EG