CREATE TABLE Driver
(
  vendor_ID VARCHAR(100) NOT NULL,
  Medallion VARCHAR(100) NOT NULL,
  Hack_License VARCHAR(10) NOT NULL,
  PRIMARY KEY (Medallion, Hack_License)
);

CREATE TABLE Location
(
  Longitude FLOAT NULL,
  Latitude FLOAT NULL,
  LocationID INT NOT NULL,
  LocationName VARCHAR(100)  NULL,
  PRIMARY KEY (LocationID)
);

CREATE TABLE Trip
(
  Trip_Distance FLOAT NOT NULL,
  Pickup_DT DATE NOT NULL,
  Dropoff_DT DATE NOT NULL,
  TripID INT NOT NULL,
  Passenger_Count INT NOT NULL,
  Medallion INT NOT NULL,
  Hack_License INT NOT NULL,
  PRIMARY KEY (TripID),
  FOREIGN KEY (Medallion, Hack_License) REFERENCES Driver(Medallion, Hack_License)
);

CREATE TABLE Payment
(
  Payment_Type VARCHAR(10) NOT NULL,
  Fare_Amount FLOAT NOT NULL,
  Surcharge FLOAT NOT NULL,
  mta_tax FLOAT NOT NULL,
  PaymentID INT NOT NULL,
  Toll_amount FLOAT NOT NULL,
  TripID INT NOT NULL,
  PRIMARY KEY (PaymentID),
  FOREIGN KEY (TripID) REFERENCES Trip(TripID)
);

CREATE TABLE To_
(
  TripID INT NOT NULL,
  LocationID INT NOT NULL,
  PRIMARY KEY (TripID, LocationID),
  FOREIGN KEY (TripID) REFERENCES Trip(TripID),
  FOREIGN KEY (LocationID) REFERENCES Location(LocationID)
);

CREATE TABLE From_
(
  TripID INT NOT NULL,
  LocationID INT NOT NULL,
  PRIMARY KEY (TripID, LocationID),
  FOREIGN KEY (TripID) REFERENCES Trip(TripID),
  FOREIGN KEY (LocationID) REFERENCES Location(LocationID)
);
