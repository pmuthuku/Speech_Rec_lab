#!/bin/perl

use strict;
use warnings;

if ($#ARGV < 0){
    print "Usage: perl format_story.pl story.txt\nWrites output to STDOUT\n";
    exit;
}

open(Input, "<$ARGV[0]");

my @words;
my $i;

while(<Input>){
    
    chomp;
    $_ =~ tr/A-Z/a-z/;
    $_ =~ s/[?;:!,.'"()-_]//g;
    @words = split(/\s+/,$_);

    foreach $i(@words){
	print "$i\n";
    }
    
    
}

close(Input);
