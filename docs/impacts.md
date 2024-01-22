## Impact Statistics Variable Names
These are the impact statistics variables produced by the PyQUANT3 model.

For example,

Ck1 = baseline by mode (k)
Ck2 = scenario by mode (k)

where

k = [ 0 | 1 | 2 ]
Mode k=0 = road
Mode k=1 = bus
Mode k=2 = rail

The following is a table of all variables produced in the resulting CSV file:

|variable | description |
|---------|-------------|
|i                             | index number      |
|Ck1 [mode k]     | Baseline population count by mode |
|Ck2 [mode k]     | Scenario population count by mode |
|CkDiff [mode k]  | Ck2-Ck1, scenario population count difference by mode |
|Lk1 [mode k]     | Baseline total distance (KM) travelled by mode |
|Lk2 [mode k]     | Scenario total distance (KM) travelled by mode |
|deltaLk [mode k] | Lk2-Lk1, scenario distance (KM) difference by mode |
|scenarioLinkDepth_k [mode k] | count of number of network link changes on each mode |
|scenarioLinkKM_k [mode k]              | Total distance (KM) of all links in the network changes  |
|scenarioLinkSavedSecs_k [mode k]      | Total amount of time saved (secs) of all links in the network changes (based on existing transit time minus new link transit time)  |
| LBar_k [mode k] | average geographic distance (KM) between all pairs of nodes in scenario |
| nMinus_k [mode k] | count of number of trips which are quicker in the scenario out of all possible N x N trips e.g. count(c=Cij1 - Cij2, where c>0) |
| savedSecs_k [mode k] | total number of seconds saved in the scenario out of all possible N x N trip times e.g. sum(t=Cij1-Cij2, where t>0) |

After the impact statistics, the input data used to create the network scenario is listed.

The "net_mode", "net_i", "net_j" and "net_secs" fields code the NetworkChanges (i.e. inputs)
to the model. These four fields repeat for each change in the scenario input data.

|variable|description|
|--------|-----------|
| net_mode | 0=road, 1=bus, 2=rail |
| net_i | origin zone number |
| net_j | destination zone number |
| net_secs | new transit time between zone i and j in seconds |

For example:

2,0,1,30 = Rail link, connect zone 0 to zone 1 with a 30 second link
