#NET
./kernel 0 datasets/converted/karate.net 3 1 none 3 #> tests/kar13.tst
./kernel 0 datasets/converted/karate.net 3 0 none 3 #> tests/kar03.tst

./kernel 0 datasets/converted/football.net 3 1 none 3 #> tests/foot13.tst
./kernel 0 datasets/converted/football.net 3 0 none 3 #> tests/foot03.tst

./kernel 0 datasets/converted/lesmiserables.net 3 1 none 3 #> tests/mis13.tst
./kernel 0 datasets/converted/lesmiserables.net 3 0 none 3 #> tests/mis03.tst

./kernel 0 datasets/converted/4adjnoun.net 3 1 none 3 #> tests/adj13.tst
./kernel 0 datasets/converted/4adjnoun.net 3 0 none 3 #> tests/adj03.tst

./kernel 0 datasets/converted/5powergrid.net 3 1 none 3 #> tests/power13.tst
./kernel 0 datasets/converted/5powergrid.net 3 0 none 3 #> tests/power03.tst

./kernel 0 datasets/converted/4internet.net 3 1 none 3 #> tests/inter13.tst
./kernel 0 datasets/converted/4internet.net 3 0 none 3 #> tests/inter03.tst

#NMI
./kernel 0 datasets/true-data/karate/karate_edges_77.txt 2 1 datasets/true-data/karate/groups.txt 3 #> tests/karate_true13.tst
./kernel 0 datasets/true-data/karate/karate_edges_77.txt 2 0 datasets/true-data/karate/groups.txt 3 #> tests/karate_true03.tst

./kernel 0 datasets/true-data/grass_web/grass_web.pairs 2 1 datasets/true-data/grass_web/grass_web.labels 3 #> tests/grass_true13.tst
./kernel 0 datasets/true-data/grass_web/grass_web.pairs 2 0 datasets/true-data/grass_web/grass_web.labels 3 #> tests/grass_true03.tst

./kernel 0 datasets/true-data/terrorists/terrorist.pairs 2 1 datasets/true-data/terrorists/terrorist.groups 3 #> tests/terrorist_true13.tst
./kernel 0 datasets/true-data/terrorists/terrorist.pairs 2 0 datasets/true-data/terrorists/terrorist.groups 3 #> tests/terrorist_true03.tst 

./kernel 0 datasets/true-data/email/email-Eu-core.txt 2 1 datasets/true-data/email/email-Eu-core-department-labels.txt 3 #> tests/email_true13.tst
./kernel 0 datasets/true-data/email/email-Eu-core.txt 2 0 datasets/true-data/email/email-Eu-core-department-labels.txt 3 #> tests/email_true03.tst

#Medium 
 ./kernel 0 datasets/com-dblp.ungraph.txt 2 1 none 3 #> tests/dblp13.tst
 ./kernel 0 datasets/com-dblp.ungraph.txt 2 0 none 3 #> tests/dblp03.tst

 ./kernel 0 datasets/com-amazon.ungraph.txt 2 1 none 3 #> tests/amazon13.tst
 ./kernel 0 datasets/com-amazon.ungraph.txt 2 0 none 3 #> tests/amazon03.tst

 #Big
#./kernel 0 datasets/com-youtube.ungraph.txt 2 1 none 3 > tests/youtube13.tst
#./kernel 0 datasets/com-youtube.ungraph.txt 2 0 none 3 > tests/youtube03.tst