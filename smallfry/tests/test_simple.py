import smallfry
import pytest

def test_bad():
	assert(False, "I told you I would fail.")

def test_good():
	assert(True, "I told you I would pass.")


