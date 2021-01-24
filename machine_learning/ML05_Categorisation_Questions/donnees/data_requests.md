I downloaded data at https://data.stackexchange.com/stackoverflow/query/new on Friday 24th July.
I did 10 requests, due to a limit of 50,000 rows by request.

The first request is:
SELECT TOP 50000 Id, Body, Title, Tags, Score, ViewCount
FROM Posts
WHERE Tags IS NOT NULL
ORDER BY Score DESC, ViewCount ASC, Id DESC

The next requests depend on *offset_number*:
SELECT Id, Body, Title, Tags, Score, ViewCount
FROM Posts
WHERE Tags IS NOT NULL
ORDER BY Score DESC, ViewCount ASC, Id DESC
OFFSET *offset_number* ROWS
FETCH NEXT 50000 ROWS ONLY

with an *offset_number* of 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000 and 450000.

