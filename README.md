# Plantdisease_tensorflow_fastAPI_tfserving_Docker
This is a project based on diseased plant leaf images. The data is given from
this link: https://www.kaggle.com/emmarex/plantdisease

In this project different library are used like tensorflow, scikit-learn, numpy,
 matplotlib, os. At first, it is tried to use different ways of loading and getting data 
in tensorflow and show how they work.

At the first part, only potato diseases are used which consist of only three classes and then
tomato, pepper and potato are used all of them together. The Model is built once using Sequential and
once by Functional API.

The data are uploaded on GoogleDrive and then mounted on Google Colaboratory.

In tensorflow_fastapi folder you can find file of fastapi and html and css files
and the model saved of potato_disease.ipynb in which by run it and giving one photo
, you receive predicted label and confidence of it.

In tfserving_fatAPI_Docker folder, you can find docker compose file. By runing docker-compose up
and docker push abzaman/tfsfastapi:tagname, you have the program ready to use and
by uploading one photo, you receive label and confidence of it.
you can pull image docker under this url:
https://hub.docker.com/repository/docker/abzaman/tfsfastapi

![00d8f10f-5038-4e0f-bb58-0b885ddc0cc5___RS_Early B 8722](https://user-images.githubusercontent.com/80074373/139596686-e621035e-db30-48c7-96ed-d891b9bc269c.JPG)

