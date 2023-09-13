What is the total distance traveled during January 2013, as well as the average distance per trip?
SELECT
  EXTRACT(DAY FROM Pickup_DT) AS DAY,
  COUNT(*) AS rides,
  SUM(Trip_Distance) AS total_distance,
  SUM(Trip_Distance) / COUNT(*) AS average_per_ride
FROM Trip
WHERE Pickup_DT BETWEEN '2013-01-13T18:57:00’
 AND '2013-01-16T11:04:00'
GROUP BY DAY
ORDER BY 1 LIMIT 100;

  
## What are the busiest hours of the day?
  
SELECT
EXTRACT(HOUR FROM Pickup_DT) AS HOUR,
COUNT(*),
SUM(Trip_Distance) AS total_distance
FROM Trip
WHERE Pickup_DT BETWEEN '2013-01-13T18:57:00' AND '2013-01-16T11:04:00'
GROUP BY HOUR
ORDER BY COUNT(*) DESC LIMIT 100;

## Correlation between the number of passengers and the distance of the trip:

SELECT Passenger_Count, YEAR( Pickup_DT ) AS YEAR, ROUND( Trip_Distance ) AS distance, COUNT( * )
FROM Trip
GROUP BY Passenger_Count, YEAR, distance
ORDER BY YEAR, COUNT( * ) DESC
LIMIT 0 , 30

## Shows about us the top 10 neighborhood that have most frequent pickups:
SELECT Trip_Distance, COUNT( * ) AS count
FROM Trip
GROUP BY Trip_Distance
ORDER BY count DESC


