#!/bin/perl

use strict;
use warnings;

my $inserts = 0;
my $subs = 0;
my $dels = 0;
my @numbers;
my $total_errs;

while(<STDIN>){
    chomp;
    @numbers = split(/\s+/,$_);
    $inserts = $inserts + $numbers[0];
    $subs = $subs + $numbers[1];
    $dels = $dels + $numbers[2];
}

print "Insertions:\t $inserts\n";
print "Substitutions:\t $subs\n";
print "Deletions:\t $dels\n";

$total_errs = $inserts + $subs + $dels;
print "Total errors:\t $total_errs\n";

