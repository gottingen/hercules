@test
def test_isdigit():
    assert "0".isdigit() == True
    assert "".isdigit() == False
    assert "a".isdigit() == False
    assert "2829357".isdigit() == True
    assert "kshfkjhe".isdigit() == False
    assert "9735g385497".isdigit() == False


@test
def test_islower():
    assert "".islower() == False
    assert "a".islower() == True
    assert "A".islower() == False
    assert "5".islower() == False
    assert "ahuiuej".islower() == True
    assert "AhUiUeJ".islower() == False
    assert "9735g385497".islower() == True
    assert "9735G385497".islower() == False


@test
def test_isupper():
    assert "".isupper() == False
    assert "a".isupper() == False
    assert "A".isupper() == True
    assert "5".isupper() == False
    assert ".J, U-I".isupper() == True
    assert "AHUIUEJ".isupper() == True
    assert "AhUiUeJ".isupper() == False
    assert "9735g385497".isupper() == False
    assert "9735G385497".isupper() == True


@test
def test_isalnum():
    assert "".isalnum() == False
    assert "a".isalnum() == True
    assert "5".isalnum() == True
    assert ",".isalnum() == False
    assert "H6".isalnum() == True
    assert ".J, U-I".isalnum() == False
    assert "A4kki83UE".isalnum() == True
    assert "AhUiUeJ".isalnum() == True
    assert "973 g38597".isalnum() == False
    assert "9735G3-5497".isalnum() == False


@test
def test_isalpha():
    assert "".isalpha() == False
    assert "a".isalpha() == True
    assert "5".isalpha() == False
    assert ",".isalpha() == False
    assert "Hh".isalpha() == True
    assert ".J, U-I".isalpha() == False
    assert "A4kki83UE".isalpha() == False
    assert "AhUiUeJ".isalpha() == True
    assert "973 g38597".isalpha() == False
    assert "9735G3-5497".isalpha() == False


@test
def test_isspace():
    assert "".isspace() == False
    assert " ".isspace() == True
    assert "5 ".isspace() == False
    assert "\t\n\r ".isspace() == True
    assert "\t ".isspace() == True
    assert "\t\ngh\r ".isspace() == False
    assert "A4kki 3UE".isspace() == False


@test
def test_istitle():
    assert "".istitle() == False
    assert " ".istitle() == False
    assert "I ".istitle() == True
    assert "IH".istitle() == False
    assert "Ih".istitle() == True
    assert "Hter Hewri".istitle() == True
    assert "Kweiur oiejf".istitle() == False


@test
def test_capitalize():
    assert " hello ".capitalize() == " hello "
    assert "Hello ".capitalize() == "Hello "
    assert "hello ".capitalize() == "Hello "
    assert "aaaa".capitalize() == "Aaaa"
    assert "AaAa".capitalize() == "Aaaa"


@test
def test_isdecimal():
    assert "".isdecimal() == False
    assert "a".isdecimal() == False
    assert "0".isdecimal() == True
    assert "\xbc".isdecimal() == False
    assert "0123456789".isdecimal() == True
    assert "0123456789a".isdecimal() == False


@test
def test_lower():
    assert "HeLLo".lower() == "hello"
    assert "hello".lower() == "hello"
    assert "HELLO".lower() == "hello"
    assert "HEL _ LO".lower() == "hel _ lo"


@test
def test_upper():
    assert "HeLLo".upper() == "HELLO"
    assert "hello".upper() == "HELLO"
    assert "HELLO".upper() == "HELLO"
    assert "HEL _ LO".upper() == "HEL _ LO"


@test
def test_isascii():
    assert "".isascii() == True
    assert "\x00".isascii() == True
    assert "\x7f".isascii() == True
    assert "\x00\x7f".isascii() == True
    assert "\x80".isascii() == False
    assert "строка".isascii() == False
    assert "\xe9".isascii() == False


@test
def test_casefold():
    assert "".casefold() == ""
    assert "HeLLo".casefold() == "hello"
    assert "hello".casefold() == "hello"
    assert "HELLO".casefold() == "hello"
    assert "HEL _ LO".casefold() == "hel _ lo"


@test
def test_swapcase():
    assert "".swapcase() == ""
    assert "HeLLo cOmpUteRs".swapcase() == "hEllO CoMPuTErS"
    assert "H.e_L,L-o cOmpUteRs".swapcase() == "h.E_l,l-O CoMPuTErS"


@test
def test_title():
    assert "".title() == ""
    assert " hello ".title() == " Hello "
    assert "hello ".title() == "Hello "
    assert "Hello ".title() == "Hello "
    assert "fOrMaT thIs aS titLe String".title() == "Format This As Title String"
    assert "fOrMaT,thIs-aS*titLe;String".title() == "Format,This-As*Title;String"
    assert "getInt".title() == "Getint"


@test
def test_isnumeric():
    assert "".isdecimal() == False
    assert "a".isdecimal() == False
    assert "0".isdecimal() == True
    assert "\xbc".isdecimal() == False
    assert "0123456789".isdecimal() == True
    assert "0123456789a".isdecimal() == False


@test
def test_ljust():
    assert "abc".ljust(10, " ") == "abc       "
    assert "abc".ljust(6, " ") == "abc   "
    assert "abc".ljust(3, " ") == "abc"
    assert "abc".ljust(2, " ") == "abc"
    assert "abc".ljust(10, "*") == "abc*******"


@test
def test_rjust():
    assert "abc".rjust(10, " ") == "       abc"
    assert "abc".rjust(6, " ") == "   abc"
    assert "abc".rjust(3, " ") == "abc"
    assert "abc".rjust(2, " ") == "abc"
    assert "abc".rjust(10, "*") == "*******abc"


@test
def test_center():
    assert "abc".center(10, " ") == "   abc    "
    assert "abc".center(6, " ") == " abc  "
    assert "abc".center(3, " ") == "abc"
    assert "abc".center(2, " ") == "abc"
    assert "abc".center(10, "*") == "***abc****"


@test
def test_zfill():
    assert "123".zfill(2) == "123"
    assert "123".zfill(3) == "123"
    assert "123".zfill(4) == "0123"
    assert "+123".zfill(3) == "+123"
    assert "+123".zfill(4) == "+123"
    assert "+123".zfill(5) == "+0123"
    assert "-123".zfill(3) == "-123"
    assert "-123".zfill(4) == "-123"
    assert "-123".zfill(5) == "-0123"
    assert "".zfill(3) == "000"
    assert "34".zfill(1) == "34"
    assert "34".zfill(4) == "0034"
    assert "1+2".zfill(5) == "001+2"
    assert "+".zfill(10) == "+000000000"
    assert "-".zfill(10) == "-000000000"


@test
def test_count():
    assert "aaa".count("a", 0, len("aaa")) == 3
    assert "aaa".count("b", 0, len("aaa")) == 0
    assert "aaa".count("a", 1, len("aaa")) == 2
    assert "aaa".count("a", 10, len("aaa")) == 0
    assert "aaa".count("a", -1, len("aaa")) == 1
    assert "aaa".count("a", 0, 1) == 1
    assert "aaa".count("a", 0, 10) == 3
    assert "aaa".count("a", 0, -1) == 2
    assert "aaa".count("aa") == 1
    assert "ababa".count("aba") == 1
    assert "abababa".count("aba") == 2
    assert "abababa".count("abab") == 1


@test
def test_find():
    assert "abcdefghiabc".find("abc", 0, len("abcdefghiabc")) == 0
    assert "abcdefghiabc".find("abc") == 0
    assert "abcdefghiabc".find("abc", 1, len("abcdefghiabc")) == 9
    assert "abcdefghiabc".find("def", 4, len("abcdefghiabc")) == -1
    assert "abcdefghiabc".find("abcdef", 0, len("abcdefghiabc")) == 0
    assert "abcdefghiabc".find("abcdef") == 0
    assert "abcdefghiabc".find("hiabc", 1, len("abcdefghiabc")) == 7
    assert "abcdefghiabc".find("defgh", 4, len("abcdefghiabc")) == -1
    assert "rrarrrrrrrrra".find("a", 0, len("rrarrrrrrrrra")) == 2
    assert "rrarrrrrrrrra".find("a", 4, len("rrarrrrrrrrra")) == 12
    assert "rrarrrrrrrrra".find("a", 4, 6) == -1
    assert "abc".find("", 0, len("abc")) == 0
    assert "abc".find("", 3, len("abc")) == 3
    assert "abc".find("", 4, len("abc")) == -1


@test
def test_rfind():
    assert "abcdefghiabc".rfind("abc", 0, len("abcdefghiabc")) == 9
    assert "abcdefghiabc".rfind("", 0, len("abcdefghiabc")) == 12
    assert "abcdefghiabc".rfind("abcd", 0, len("abcdefghiabc")) == 0
    assert "abcdefghiabc".rfind("abcz", 0, len("abcdefghiabc")) == -1
    assert "abcdefghiabc".rfind("abc") == 9
    assert "abcdefghiabc".rfind("") == 12
    assert "abcdefghiabc".rfind("abcd") == 0
    assert "abcdefghiabc".rfind("abcz") == -1
    assert "rrarrrrrrrrra".rfind("a", 0, len("rrarrrrrrrrra")) == 12
    assert "rrarrrrrrrrra".rfind("a", 4, len("rrarrrrrrrrra")) == 12
    assert "rrarrrrrrrrra".rfind("a", 4, 6) == -1
    assert "abc".rfind("", 0, len("abc")) == 3
    assert "abc".rfind("", 3, len("abc")) == 3
    assert "abc".rfind("", 4, len("abc")) == -1


@test
def test_isidentifier():
    assert "a".isidentifier() == True
    assert "Z".isidentifier() == True
    assert "_".isidentifier() == True
    assert "b0".isidentifier() == True
    assert "bc".isidentifier() == True
    assert "b_".isidentifier() == True
    assert " ".isidentifier() == False
    assert "3t".isidentifier() == False
    assert "_gth_45".isidentifier() == True


@test
def test_isprintable():
    assert "".isprintable() == True
    assert '"'.isprintable() == True
    assert "'".isprintable() == True
    assert " ".isprintable() == True
    assert "abcdef".isprintable() == True
    assert "0123456789".isprintable() == True
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ".isprintable() == True
    assert "abcdefghijklmnopqrstuvwxyz".isprintable() == True
    assert "!#$%&()*+,-./:;?@[\\]^_`{|}~".isprintable() == True
    assert "abcdef\n".isprintable() == False


@test
def test_lstrip():
    assert "".lstrip() == ""
    assert "   ".lstrip() == ""
    assert "   hello   ".lstrip("") == "hello   "
    assert " \t\n\rabc \t\n\r".lstrip("") == "abc \t\n\r"
    assert "xyzzyhelloxyzzy".lstrip("xyz") == "helloxyzzy"


@test
def test_rstrip():
    assert "".rstrip() == ""
    assert "   ".rstrip() == ""
    assert "   hello   ".rstrip("") == "   hello"
    assert " \t\n\rabc \t\n\r".rstrip("") == " \t\n\rabc"
    assert "xyzzyhelloxyzzy".rstrip("xyz") == "xyzzyhello"


@test
def test_strip():
    assert "".strip() == ""
    assert "   ".strip() == ""
    assert "   hello   ".strip("") == "hello"
    assert "   hello   ".strip() == "hello"
    assert " \t\n\rabc \t\n\r".strip() == "abc"
    assert "xyzzyhelloxyzzy".strip("xyz") == "hello"
    assert "hello".strip("xyz") == "hello"
    assert "mississippi".strip("mississippi") == ""
    assert "mississippi".strip("i") == "mississipp"


@test
def test_partition():
    assert "hello".partition("l") == ("he", "l", "lo")
    assert "this is the partition method".partition("ti") == (
        "this is the par",
        "ti",
        "tion method",
    )
    assert "http://www.seq.org".partition("://") == ("http", "://", "www.seq.org")
    assert "http://www.seq.org".partition("?") == ("http://www.seq.org", "", "")
    assert "http://www.seq.org".partition("http://") == ("", "http://", "www.seq.org")
    assert "http://www.seq.org".partition("org") == ("http://www.seq.", "org", "")


@test
def test_rpartition():
    assert "hello".rpartition("l") == ("hel", "l", "o")
    assert "this is the rpartition method".rpartition("ti") == (
        "this is the rparti",
        "ti",
        "on method",
    )
    assert "http://www.seq.org".rpartition("://") == ("http", "://", "www.seq.org")
    assert "http://www.seq.org".rpartition("?") == ("", "", "http://www.seq.org")
    assert "http://www.seq.org".rpartition("http://") == ("", "http://", "www.seq.org")
    assert "http://www.seq.org".rpartition("org") == ("http://www.seq.", "org", "")


@test
def test_split():
    assert "  h    l \t\n l   o ".split() == ["h", "l", "l", "o"]
    assert "  h    l \t\n l   o ".split(None, 2) == ["h", "l", "l   o "]
    assert "  h    l \t\n l   o ".split(None, 0) == ["h    l \t\n l   o "]
    assert not "".split()
    assert not "   ".split()
    assert "h l l o".split(" ", -1) == ["h", "l", "l", "o"]
    assert "a|b|c|d".split("|", -1) == ["a", "b", "c", "d"]
    assert "h l l o".split(" ") == ["h", "l", "l", "o"]
    assert "a|b|c|d".split("|") == ["a", "b", "c", "d"]
    assert "a|b|c|d".split("|", 0) == ["a|b|c|d"]
    assert "abcd".split("|", -1) == ["abcd"]
    assert "".split("|", -1) == [""]
    assert "endcase |".split("|", -1) == ["endcase ", ""]
    assert "| startcase".split("|", -1) == ["", " startcase"]
    assert "|bothcase|".split("|", -1) == ["", "bothcase", ""]
    assert "abbbc".split("bb", -1) == ["a", "bc"]
    assert "aaa".split("aaa", -1) == ["", ""]
    assert "aaa".split("aaa", 0) == ["aaa"]
    assert "abbaab".split("ba", -1) == ["ab", "ab"]
    assert "aa".split("aaa", -1) == ["aa"]
    assert "Abbobbbobb".split("bbobb", -1) == ["A", "bobb"]
    assert "AbbobbBbbobb".split("bbobb", -1) == ["A", "B", ""]
    assert ("a|" * 20)[:-1].split("|", -1) == ["a"] * 20
    assert ("a|" * 20)[:-1].split("|", 15) == ["a"] * 15 + ["a|a|a|a|a"]
    assert "a|b|c|d".split("|", 1) == ["a", "b|c|d"]
    assert "a|b|c|d".split("|", 2) == ["a", "b", "c|d"]
    assert "a|b|c|d".split("|", 3) == ["a", "b", "c", "d"]
    assert "a|b|c|d".split("|", 4) == ["a", "b", "c", "d"]
    assert "a||b||c||d".split("|", 2) == ["a", "", "b||c||d"]


@test
def test_rsplit():
    assert "  h    l \t\n l   o ".rsplit() == ["h", "l", "l", "o"]
    assert "  h    l \t\n l   o ".rsplit(None, 2) == ["  h    l", "l", "o"]
    assert "  h    l \t\n l   o ".rsplit(None, 0) == ["  h    l \t\n l   o"]
    assert not "".rsplit()
    assert not "   ".rsplit()
    assert "a|b|c|d".rsplit("|", -1) == ["a", "b", "c", "d"]
    assert "a|b|c|d".rsplit("|") == ["a", "b", "c", "d"]
    assert "a|b|c|d".rsplit("|", 1) == ["a|b|c", "d"]
    assert "a|b|c|d".rsplit("|", 2) == ["a|b", "c", "d"]
    assert "a|b|c|d".rsplit("|", 3) == ["a", "b", "c", "d"]
    assert "a|b|c|d".rsplit("|", 4) == ["a", "b", "c", "d"]
    assert "a|b|c|d".rsplit("|", 0) == ["a|b|c|d"]
    assert "a||b||c||d".rsplit("|", 2) == ["a||b||c", "", "d"]
    assert "abcd".rsplit("|", -1) == ["abcd"]
    assert "".rsplit("|", -1) == [""]
    assert "endcase |".rsplit("|", -1) == ["endcase ", ""]
    assert "| startcase".rsplit("|", -1) == ["", " startcase"]
    assert "|bothcase|".rsplit("|", -1) == ["", "bothcase", ""]
    # assert 'a\x00\x00b\x00c\x00d'.rsplit('\x00', -1)
    assert "abbbc".rsplit("bb", -1) == ["ab", "c"]
    assert "aaa".rsplit("aaa", -1) == ["", ""]
    assert "aaa".rsplit("aaa", 0) == ["aaa"]
    assert "abbaab".rsplit("ba", -1) == ["ab", "ab"]
    assert "aa".rsplit("aaa", -1) == ["aa"]
    assert "bbobbbobbA".rsplit("bbobb", -1) == ["bbob", "A"]
    assert "bbobbBbbobbA".rsplit("bbobb", -1) == ["", "B", "A"]
    assert ("aBLAH" * 20)[:-4].rsplit("BLAH", -1) == ["a"] * 20
    assert ("a|" * 20)[:-1].rsplit("|", 15) == ["a|a|a|a|a"] + ["a"] * 15
    assert "a||b||c||d".rsplit("|", 2) == ["a||b||c", "", "d"]


@test
def test_splitlines():
    assert "\n\nasdf\nsadf\nsdf\n".splitlines(False) == ["", "", "asdf", "sadf", "sdf"]
    assert "\n\nasdf\nsadf\nsdf\n".splitlines() == ["", "", "asdf", "sadf", "sdf"]
    assert "abc\ndef\n\rghi".splitlines(False) == ["abc", "def", "", "ghi"]
    assert "abc\ndef\n\r\nghi".splitlines(False) == ["abc", "def", "", "ghi"]
    assert "abc\ndef\r\nghi".splitlines(False) == ["abc", "def", "ghi"]
    assert "abc\ndef\r\nghi\n".splitlines(False) == ["abc", "def", "ghi"]
    assert "abc\ndef\r\nghi\n\r".splitlines(False) == ["abc", "def", "ghi", ""]
    assert "\nabc\ndef\r\nghi\n\r".splitlines(False) == ["", "abc", "def", "ghi", ""]
    assert "\nabc\ndef\r\nghi\n\r".splitlines(True) == [
        "\n",
        "abc\n",
        "def\r\n",
        "ghi\n",
        "\r",
    ]
    assert "abc\ndef\r\nghi\n".splitlines(True) == ["abc\n", "def\r\n", "ghi\n"]


@test
def test_startswith():
    assert "hello".startswith("he", 0, len("hello")) == True
    assert "hello".startswith("hello", 0, len("hello")) == True
    assert "hello".startswith("hello world", 0, len("hello")) == False
    assert "hello".startswith("", 0, len("hello")) == True
    assert "hello".startswith("ello", 0, len("hello")) == False
    assert "hello".startswith("he") == True
    assert "hello".startswith("hello") == True
    assert "hello".startswith("hello world") == False
    assert "hello".startswith("") == True
    assert "hello".startswith("ello") == False
    assert "hello".startswith("ello", 1, len("hello")) == True
    assert "hello".startswith("o", 4, len("hello")) == True
    assert "hello".startswith("o", 5, len("hello")) == False
    assert "hello".startswith("lo", 3, len("hello")) == True
    assert "hello".startswith("", 5, len("hello")) == True
    assert "hello".startswith("lo", 6, len("hello")) == False
    assert "helloworld".startswith("lowo", 3, len("helloworld")) == True
    assert "helloworld".startswith("lowo", 3, 7) == True
    assert "helloworld".startswith("lowo", 3, 6) == False
    assert "".startswith("", 0, 1) == True
    assert "".startswith("", 0, 0) == True
    assert "".startswith("", 1, 0) == False
    assert "hello".startswith("he", 0, -1) == True
    assert "hello".startswith("hello", 0, -1) == False
    assert "hello".startswith("he", 0, -3) == True
    assert "hello".startswith("ello", -4, len("hello")) == True
    assert "hello".startswith("ello", -5, len("hello")) == False
    assert "hello".startswith("", -3, -3) == True
    assert "hello".startswith("o", -1, len("hello")) == True


@test
def test_endswith():
    assert "hello".endswith("lo", 0, len("hello")) == True
    assert "hello".endswith("he", 0, len("hello")) == False
    assert "hello".endswith("", 0, len("hello")) == True
    assert "hello".endswith("hello world", 0, len("hello")) == False
    assert "hello".endswith("lo") == True
    assert "hello".endswith("he") == False
    assert "hello".endswith("") == True
    assert "hello".endswith("hello world") == False
    assert "helloworld".endswith("worl", 0, len("hello")) == False
    assert "helloworld".endswith("worl", 3, 9) == True
    assert "helloworld".endswith("world", 3, 12) == True
    assert "helloworld".endswith("lowo", 1, 7) == True
    assert "helloworld".endswith("lowo", 2, 7) == True
    assert "helloworld".endswith("lowo", 3, 7) == True
    assert "helloworld".endswith("lowo", 4, 7) == False
    assert "helloworld".endswith("lowo", 3, 8) == False
    assert "ab".endswith("ab", 0, 1) == False
    assert "ab".endswith("ab", 0, 0) == False
    assert "".endswith("", 0, 1) == True
    assert "".endswith("", 0, 0) == True
    assert "".endswith("", 1, 0) == False
    assert "hello".endswith("lo", -2, len("hello")) == True
    assert "hello".endswith("he", -2, len("hello")) == False
    assert "hello".endswith("", -3, -3) == True
    assert "helloworld".endswith("worl", -6, len("helloworld")) == False
    assert "helloworld".endswith("worl", -5, -1) == True
    assert "helloworld".endswith("worl", -5, 9) == True
    assert "helloworld".endswith("world", -7, 12) == True
    assert "helloworld".endswith("lowo", -99, -3) == True
    assert "helloworld".endswith("lowo", -8, -3) == True
    assert "helloworld".endswith("lowo", -7, -3) == True
    assert "helloworld".endswith("lowo", 3, -4) == False
    assert "helloworld".endswith("lowo", -8, -2) == False


@test
def test_index():
    assert "abcdefghiabc".index("abc", 0, len("abcdefghiabc")) == 0
    assert "abcdefghiabc".index("abc") == 0
    assert "abcdefghiabc".index("abc", 1, len("abcdefghiabc")) == 9
    assert "abc".index("", 0, len("abc")) == 0
    assert "abc".index("", 3, len("abc")) == 3
    assert "rrarrrrrrrrra".index("a", 0, len("rrarrrrrrrrra")) == 2
    assert "rrarrrrrrrrra".index("a", 4, len("rrarrrrrrrrra")) == 12
    try:
        "abcdefghiabc".index("def", 4, len("abcdefghiabc"))
        assert False
    except ValueError:
        pass


@test
def test_rindex():
    assert "abcdefghiabc".rindex("", 0, len("abcdefghiabc")) == 12
    assert "abcdefghiabc".rindex("") == 12
    assert "abcdefghiabc".rindex("def", 0, len("abcdefghiabc")) == 3
    assert "abcdefghiabc".rindex("abc", 0, len("abcdefghiabc")) == 9
    assert "abcdefghiabc".rindex("abc", 0, -1) == 0
    assert "rrarrrrrrrrra".rindex("a", 0, len("rrarrrrrrrrra")) == 12
    assert "rrarrrrrrrrra".rindex("a", 4, len("rrarrrrrrrrra")) == 12
    try:
        "rrarrrrrrrrra".rindex("a", 4, 6)
        assert False
    except ValueError:
        pass


@test
def test_replace():
    # interleave-- default will be len(str) + 1
    assert "A".replace("", "", len("A") + 1) == "A"
    assert "A".replace("", "*", len("A") + 1) == "*A*"
    assert "A".replace("", "*1", len("A") + 1) == "*1A*1"
    assert "A".replace("", "*-#", len("A") + 1) == "*-#A*-#"
    assert "AA".replace("", "*-", len("AA") + 1) == "*-A*-A*-"
    assert "AA".replace("", "*-", -1) == "*-A*-A*-"
    assert "AA".replace("", "*-") == "*-A*-A*-"
    assert "AA".replace("", "*-", 4) == "*-A*-A*-"
    assert "AA".replace("", "*-", 3) == "*-A*-A*-"
    assert "AA".replace("", "*-", 2) == "*-A*-A"
    assert "AA".replace("", "*-", 1) == "*-AA"
    assert "AA".replace("", "*-", 0) == "AA"

    # substring deletion
    assert "A".replace("A", "", len("A") + 1) == ""
    assert "AAA".replace("A", "", len("AAA") + 1) == ""
    assert "AAA".replace("A", "", -1) == ""
    assert "AAA".replace("A", "") == ""
    assert "AAA".replace("A", "", 4) == ""
    assert "AAA".replace("A", "", 3) == ""
    assert "AAA".replace("A", "", 2) == "A"
    assert "AAA".replace("A", "", 1) == "AA"
    assert "AAA".replace("A", "", 0) == "AAA"
    assert "ABACADA".replace("A", "", len("ABACADA") + 1) == "BCD"
    assert "ABACADA".replace("A", "", -1) == "BCD"
    assert "ABACADA".replace("A", "", 5) == "BCD"
    assert "ABACADA".replace("A", "", 4) == "BCD"
    assert "ABACADA".replace("A", "", 3) == "BCDA"
    assert "ABACADA".replace("A", "", 2) == "BCADA"
    assert "ABACADA".replace("A", "", 1) == "BACADA"
    assert "ABACADA".replace("A", "", 0) == "ABACADA"
    assert "ABCAD".replace("A", "", len("ABCAD") + 1) == "BCD"
    assert "ABCADAA".replace("A", "", len("ABCADAA") + 1) == "BCD"
    assert "BCD".replace("A", "", len("BCD") + 1) == "BCD"
    assert ("^" + ("A" * 1000) + "^").replace("A", "", 999) == "^A^"
    assert "the".replace("the", "", len("the") + 1) == ""
    assert "theater".replace("the", "", len("theater") + 1) == "ater"
    assert "thethe".replace("the", "", len("thethe") + 1) == ""
    assert "thethethethe".replace("the", "", len("thethethethe") + 1) == ""
    assert "theatheatheathea".replace("the", "", len("theatheatheathea") + 1) == "aaaa"
    assert "that".replace("the", "", len("that") + 1) == "that"
    assert (
        "here and there".replace("the", "", len("here and there") + 1) == "here and re"
    )
    assert (
        "here and there and there".replace(
            "the", "", len("here and there and there") + 1
        )
        == "here and re and re"
    )
    assert "here and there and there".replace("the", "", -1) == "here and re and re"
    assert "here and there and there".replace("the", "", 3) == "here and re and re"
    assert "here and there and there".replace("the", "", 2) == "here and re and re"
    assert "here and there and there".replace("the", "", 1) == "here and re and there"
    assert (
        "here and there and there".replace("the", "", 0) == "here and there and there"
    )

    # substring replace in place
    assert (
        "Who goes there?".replace("o", "o", len("Who goes there?") + 1)
        == "Who goes there?"
    )
    assert (
        "Who goes there?".replace("o", "O", len("Who goes there?") + 1)
        == "WhO gOes there?"
    )
    assert "Who goes there?".replace("o", "O", -1) == "WhO gOes there?"
    assert "Who goes there?".replace("o", "O", 3) == "WhO gOes there?"
    assert "Who goes there?".replace("o", "O", 2) == "WhO gOes there?"
    assert "Who goes there?".replace("o", "O", 1) == "WhO goes there?"
    assert "Who goes there?".replace("o", "O", 0) == "Who goes there?"
    assert (
        "Who goes there?".replace("a", "q", len("Who goes there?") + 1)
        == "Who goes there?"
    )
    assert (
        "Who goes there?".replace("W", "w", len("Who goes there?") + 1)
        == "who goes there?"
    )
    assert (
        "WWho goes there?WW".replace("W", "w", len("WWho goes there?WW") + 1)
        == "wwho goes there?ww"
    )
    assert (
        "Who goes there?".replace("?", "!", len("Who goes there?") + 1)
        == "Who goes there!"
    )
    assert (
        "This is a tissue".replace("is", "**", len("This is a tissue") + 1)
        == "Th** ** a t**sue"
    )
    assert "This is a tissue".replace("is", "**", -1) == "Th** ** a t**sue"
    assert "This is a tissue".replace("is", "**", 4) == "Th** ** a t**sue"
    assert "This is a tissue".replace("is", "**", 3) == "Th** ** a t**sue"
    assert "This is a tissue".replace("is", "**", 2) == "Th** ** a tissue"
    assert "This is a tissue".replace("is", "**", 1) == "Th** is a tissue"
    assert "This is a tissue".replace("is", "**", 0) == "This is a tissue"
    assert "Reykjavik".replace("k", "KK", len("Reykjavik") + 1) == "ReyKKjaviKK"
    assert "Reykjavik".replace("k", "KK", -1) == "ReyKKjaviKK"
    assert "Reykjavik".replace("k", "KK", 2) == "ReyKKjaviKK"
    assert "Reykjavik".replace("k", "KK", 1) == "ReyKKjavik"
    assert "Reykjavik".replace("k", "KK", 0) == "Reykjavik"
    assert "A.B.C.".replace(".", "----", len("A.B.C.") + 1) == "A----B----C----"
    assert (
        "spam, spam, eggs and spam".replace(
            "spam", "ham", len("spam, spam, eggs and spam") + 1
        )
        == "ham, ham, eggs and ham"
    )
    assert (
        "spam, spam, eggs and spam".replace("spam", "ham", -1)
        == "ham, ham, eggs and ham"
    )
    assert (
        "spam, spam, eggs and spam".replace("spam", "ham", 4)
        == "ham, ham, eggs and ham"
    )
    assert (
        "spam, spam, eggs and spam".replace("spam", "ham", 3)
        == "ham, ham, eggs and ham"
    )
    assert (
        "spam, spam, eggs and spam".replace("spam", "ham", 2)
        == "ham, ham, eggs and spam"
    )
    assert (
        "spam, spam, eggs and spam".replace("spam", "ham", 1)
        == "ham, spam, eggs and spam"
    )
    assert (
        "spam, spam, eggs and spam".replace("spam", "ham", 0)
        == "spam, spam, eggs and spam"
    )


@test
def test_expandtabs():
    assert "abc\rab\tdef\ng\thi".expandtabs(8) == "abc\rab      def\ng       hi"
    assert "abc\rab\tdef\ng\thi".expandtabs(8) == "abc\rab      def\ng       hi"
    assert "abc\rab\tdef\ng\thi".expandtabs(4) == "abc\rab  def\ng   hi"
    assert "abc\r\nab\tdef\ng\thi".expandtabs(8) == "abc\r\nab      def\ng       hi"
    assert "abc\r\nab\tdef\ng\thi".expandtabs(4) == "abc\r\nab  def\ng   hi"
    assert "abc\r\nab\r\ndef\ng\r\nhi".expandtabs(4) == "abc\r\nab\r\ndef\ng\r\nhi"
    assert " \ta\n\tb".expandtabs(1) == "  a\n b"
    assert "\tdndhd\ty\ty\tyu\t".expandtabs(3) == "   dndhd y  y  yu "


@test
def test_translate():
    assert "I yor ge".translate({ord("g"): "w", ord("y"): "f"}) == "I for we"
    assert "abababc".translate({ord("a"): ""}) == "bbbc"
    assert "abababc".translate({ord("a"): "", ord("b"): "i"}) == "iiic"
    assert "abababc".translate({ord("a"): "", ord("b"): "i", ord("c"): "x"}) == "iiix"
    assert "abababc".translate({ord("a"): "", ord("b"): ""}) == "c"
    assert "xzx".translate({ord("z"): "yy"}) == "xyyx"
    assert "aaabbbccc".translate({ord("b"): Optional("XY"), ord("c"): None, ord("a"): Optional("")}) == "XYXYXY"

@test
def test_repr():
    assert repr("") == "''"
    assert repr("hello") == "'hello'"
    assert repr("     ") == "'     '"
    assert repr("\r\a\n\t") == "'\\r\\a\\n\\t'"


@test
def test_fstr():
    assert f"{2+2}" == "4"
    n = 42
    assert f"{n}{n}xx{n}" == "4242xx42"
    assert f"{n=}" == "n=42"
    assert f"hello {n=} world" == "hello n=42 world"


@test
def test_slice(
    s="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    indices=(0, 1, 3, 41, 0xFFFFFFFFFFF, -1, -2, -37),
):
    for start in indices:
        for stop in indices:
            for step in indices[1:]:
                L = list(s)[start:stop:step]
                assert s[start:stop:step] == "".join(L)


@test
def test_join():
    assert "".join(str(a) for a in range(0)) == ""
    assert "".join(List[str]()) == ""
    assert "a".join(str(a) for a in range(0)) == ""
    assert "a".join(List[str]()) == ""
    assert "ab".join(str(a) for a in range(999, 1000)) == "999"
    assert "ab".join(["999"]) == "999"
    assert "xyz".join(str(a) for a in range(5)) == "0xyz1xyz2xyz3xyz4"
    assert "xyz".join(["00", "1", "22", "3", "44"]) == "00xyz1xyz22xyz3xyz44"
    assert "xyz".join(iter(["00", "1", "22", "3", "44"])) == "00xyz1xyz22xyz3xyz44"
    assert "xyz".join(["00", "1", "22", "3", "44"]) == "00xyz1xyz22xyz3xyz44"
    assert "xyz".join(iter(["00", "", "22", "3", ""])) == "00xyzxyz22xyz3xyz"


@test
def test_repr():
    s = (
        "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19"
        "\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefg"
        "hijklmnopqrstuvwxyz{|}~\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91"
        "\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xff"
    )
    assert repr(s) == (
        "'\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\t\\n\\x0b\\x0c\\r\\x0e\\x0f\\x10\\x11"
        "\\x12\\x13\\x14\\x15\\x16\\x17\\x18\\x19\\x1a\\x1b\\x1c\\x1d\\x1e\\x1f !\"#$%&\\'()*+,"
        "-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\\"
        "x7f\\x80\\x81\\x82\\x83\\x84\\x85\\x86\\x87\\x88\\x89\\x8a\\x8b\\x8c\\x8d\\x8e\\x8f\\x90"
        "\\x91\\x92\\x93\\x94\\x95\\x96\\x97\\x98\\x99\\x9a\\x9b\\x9c\\x9d\\x9e\\x9f\\xa0\\xff'"
    )
    assert repr("") == "''"
    assert repr('"') == "'\"'"
    assert repr("'") == '"\'"'
    assert repr("\"'") == "'\"\\''"


test_isdigit()
test_islower()
test_isupper()
test_isalnum()
test_isalpha()
test_isspace()
test_istitle()
test_capitalize()
test_isdecimal()
test_lower()
test_upper()
test_isascii()
test_casefold()
test_swapcase()
test_title()
test_isnumeric()
test_ljust()
test_rjust()
test_center()
test_zfill()
test_count()
test_find()
test_rfind()
test_isidentifier()
test_isprintable()
test_lstrip()
test_rstrip()
test_strip()
test_partition()
test_rpartition()
test_split()
test_rsplit()
test_splitlines()
test_startswith()
test_endswith()
test_index()
test_rindex()
test_replace()
test_expandtabs()
test_translate()
test_repr()
test_fstr()
test_slice()
test_join()
test_repr()
