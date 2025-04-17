# mldeployment Housing price prediction model


# Go to the directory in terminal
cd app_housing

# Build Docker image
docker build -t ml-housing-model .

# Run Docker container
docker run -p 8000:9000 ml-housing-model

# Test the API in new terminal

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 1.0]]}'

expected output
{
    "prediction": [
        11917308.2
    ]
}
# The API can handle multiple sample

curl -X POST http://localhost:8000/predict \\ \
     -H "Content-Type: application/json" \\ \
     -d '{"features": [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 1.0],[8960, 4, 4, 4, 1, 0, 0, 0, 1, 3, 0, 1.0]]}'

expected output
{
    "prediction": [
        11917308.2,
        10627429.4
    ]
}

# Using the API
When using the API ensure that you have exactly 12 input feature or it will throw the error.\
The number are related to column name in order. \
When the feature is a yes or no, input 1 for yes and 0 for no. \
area - Area of a House in square feet	\
bedrooms - Number of House Bedrooms	\
bathrooms - Number of Bathrooms	\
stories	- Number of House Stories \
mainroad - Weather connected to Main Road (0,1)	\
guestroom - Weather has a guest room (0,1) \
basement - Weather has a basement (0,1)	\
hotwaterheating	- Weather has a hotwater heater (0,1) \
airconditioning	- Weather has an airconditioning (0,1)\
parking	- Number of House Parkings \
prefarea - 	indicating whether the house is located in a preferred neighborhood or locality (0,1) \
furnishingstatus - Furnishing status of the House (0 - unfurnished, 0.5 - semi-furnished, 1 - furnished)

