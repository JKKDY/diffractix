from diffractix.elements import Space


if __name__ == "__main__":
    x = Space(d=0.3, label="Slider").variable()
    print(x.has_variable_length)