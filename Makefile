.PHONY: prefix all

all: prefix
	echo "world"

install:
	pip install -r requirements.txt

prefix:
	echo "hello"