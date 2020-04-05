---
layout: post
title: "Implementing natural numbers in OCaml"
date: 2021-04-06 08:00:00 +0000
categories: programming
---

In this post we're going to implement natural numbers (positive integers) in [OCaml](https://ocaml.org/) to see how we can define numbers from first
principle, namely without using OCaml's built in `Integer` type. We'll then write a simple UI so that we have a basic (but inefficient) calculator.


## Definition

We'll start with a recursive definition of natural numbers:

$$
n \in \mathcal{N} \iff n = \begin{cases}0 \\ S(m) \hspace{5mm} \text{for }m \in \mathcal{N}
\end{cases}
$$

With $$S(m)$$ the [successor function](https://en.wikipedia.org/wiki/Successor_function), which is a function that simply returns the next natural number. This means that a natural number is either $$0$$ or the successor of another natural number. For example $$0$$ is a natural number (the first case in the definition), but $$1$$ is also a natural number, as it's the successor of $$0$$ (you would write $$1=S(0)$$). 2 can then be written as $$2 = S(S(0))$$ , and so on. By using recursion (the definition of a natural number includes another natural number) we can "bootstrap" building numbers without using many other definitions.


We now write this definition as a type in OCaml, which looks a lot like the mathematical definition above:


```ocaml
type nat =
  | Zero
  | Succ of nat
```

The vertical lines denote the two cases. Here you would write 1 as `Succ Zero`, 2 as `Succ Succ Zero`, and so on.


However we haven't said what these numbers are (what _is_ zero? What _are_ numbers? ). To do
that we need to define how they act.

## Some operators

We'll start off by defining how we can increment and decrement them.

```ocaml
let incr n =
  Succ n

let decr n =
  match n with
  | Zero -> Zero
  | Succ nn -> nn
```

The increment function simply adds a `Succ` before the number, so this corresonds to adding 1. So `incr (Succ Zero)` returns `Succ Succ Zero`. The decrement function checks whether the number `n` is `Zero` or the successor of a number. In the first case it simply returns `Zero` (So this means that `decr Zero` returns `Zero`. However this could be extended to include negative numbers). In the second case the function returns the number that precedes it. So `decr (Succ Succ Succ Zero)` returns `Succ Succ Zero`.

### Addition

We can now define addition as a recursive function which we denote by `++` (in OCaml we define [infix operators](https://en.wikipedia.org/wiki/Infix_notation) using parentheses). So the addition function takes two elements `n` and `m` of type `nat` and returns an element of type `nat`. Note the `rec` added before the function name which means that it's recursive.

```ocaml
let rec (++) n m =
  match m with
  | Zero -> n
  | Succ mm -> (Succ n) ++ mm
```
Because we defined the function to be an infix operator we put it in between the arguments (ex: `Zero ++ (Succ Zero)`). This function checks whether `m` is `Zero` or the successor of a number. If it's a successor of `mm` it returns the sum of `mm` and `Succ n`.

Let check that this definition behaves correctly by calculating 1+1 which we write as `(Succ Zero) ++ (Succ Zero)`. The first call to the function finds that the second argument is the successor of `Zero`, so returns the sum `(Succ Succ Zero) ++ Zero`. This calls the functions a second time which finds that the second argument is `Zero`. As a result the function return `Succ Succ Zero` which is 2 !

So in summary 1+1 is written as `(Succ Zero) ++ (Succ Zero)` = `(Succ Succ Zero) ++ Zero` = `Succ Succ Zero`. Math still works!

### Subtraction

We now define subtraction:

```ocaml
let rec (--) n m =
  match m with
  | Zero -> n
  | Succ mm -> (decr n) -- mm
```

This decrements both arguments until the second one is Zero. Note that if `m` is bigger than `n` then `n -- m` will still equal `Zero`.

### Multiplication

Moving on, we define multiplication:

```ocaml
let (+*) n m =
  let rec aux n m acc =
    match m with
    | Zero -> acc
    | Succ mm -> aux n mm (n ++ acc)
  in
  aux n m Zero
```

Here we use an auxiliary function which builds up the result in the accumulator `acc` by adding `n` to it `m` times. So applying this function to $$3$$ and $$2$$ gives: $$3*2 = 3*1 + 3 = 3*0 + 6 = 6$$. And in code this is:

- `(Succ (Succ (Succ Zero))) +* (Succ (Succ Zero))`
- Which returns `((Succ (Succ (Succ Zero))) +* (Succ Zero)) ++ (Succ (Succ (Succ Zero)))`
- Which returns `((Succ (Succ (Succ Zero))) +* Zero) ++ (Succ (Succ (Succ (Succ (Succ (Succ Zero)))))) `
- which returns `(Succ (Succ (Succ (Succ (Succ (Succ Zero))))))` (namely $$6$$)

### Division

We also define the 'strictly less than' operator which we then use to define integer division.

```ocaml
let rec (<<) n m =
  match (n, m) with
  | (p, Zero) -> false
  | (Zero, q) -> true
  | (p, q) -> (decr n) << (decr m)

let (//) n m =
  let rec aux p acc =
    let lt = p << m in
    match lt with
    | true -> acc
    | false -> aux (p -- m) (Succ acc)
  in
  aux n Zero
```

Finally we can define the modulo operator. As we use previous definitions of division, multiplication, and subtraction, this definition is abstracted away from our implementation of natural numbers.

```ocaml

let (%) n m =
  let p = n // m in
  n -- (p +* m)
```

## A basic UI

We've defined the natural numbers and the basic operators, but it's a bit unwieldy to use them in their current form. So we'll write some code to convert them to the usual number system (represented as strings) and back.


### From type `nat` to string representation

We'll write some code to convert numbers to base 10 and then represent them in the usual Arabic numerals.

```ocaml
let ten = Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ Zero)))))))))

let base10 n =
  let rec aux q acc =
    let r = q % ten in
    let p = q // ten in
    match p with
    | Zero -> r::acc
    | pp -> aux p (r::acc)
  in
  aux n []
```

This function returns a list where each element corresponds to the number of 1s, 10s, 100s etc in the number. So if `n` is `Succ Succ Succ Succ Succ Succ Succ Succ Succ Succ Succ Succ Zero` (ie: 12), then `base10 n` returns `[Succ Zero; Succ Succ Zero]`.

We then define the 10 digits (with a hack for the cases bigger than 9) and put it all together in the function `string_of_nat`.

```ocaml
let print_nat_digits = function
  | Zero -> "0"
  | Succ Zero -> "1"
  | Succ Succ Zero -> "2"
  | Succ Succ Succ Zero -> "3"
  | Succ Succ Succ Succ Zero -> "4"
  | Succ Succ Succ Succ Succ Zero -> "5"
  | Succ Succ Succ Succ Succ Succ Zero -> "6"
  | Succ Succ Succ Succ Succ Succ Succ Zero -> "7"
  | Succ Succ Succ Succ Succ Succ Succ Succ Zero -> "8"
  | Succ Succ Succ Succ Succ Succ Succ Succ Succ Zero -> "9"
  | _ -> "bigger than 9"

let string_of_nat n =
  let base_10_rep = base10 n in
  let list_strings = List.map print_nat_digits base_10_rep in
  String.concat "" list_strings
```

`string_of_nat` converts the number of type `nat` to base 10, then maps each of the list element to a string and concatenates those strings.

So `string_of_nat (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ Zero))))))))))))` returns `"12"` which is easier to read!

### From string representation to type `nat`

We then define some code to go the other way around: from string representation to natural numbers.

```ocaml
let string_to_list s =
  let rec loop acc i =
    if i = -1 then acc
    else
      loop ((String.make 1 s.[i]) :: acc) (pred i)
  in loop [] (String.length s - 1)


let nat_of_listnat l =
  let lr = List.rev l in
  let rec aux n b lr =
    match lr with
    | [] -> n
    | h::t -> aux (n ++ (b+*h)) (b+*ten) t
  in
  aux Zero (Succ Zero) lr

let nat_of_string_digits = function
  | "0" -> Zero
  | "1" -> Succ Zero
  | "2" -> Succ (Succ Zero)
  | "3" -> Succ (Succ (Succ Zero))
  | "4" -> Succ (Succ (Succ (Succ Zero)))
  | "5" -> Succ (Succ (Succ (Succ (Succ Zero))))
  | "6" -> Succ (Succ (Succ (Succ (Succ (Succ Zero)))))
  | "7" -> Succ (Succ (Succ (Succ (Succ (Succ (Succ Zero))))))
  | "8" -> Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ Zero)))))))
  | "9" -> Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ Zero))))))))
  | _ -> raise (Failure "string must be less than 10")

(* Converts string to nat *)
let nat_of_string s =
  let liststring = string_to_list s in
  let listNatbase = List.map nat_of_string_digits liststring in
  nat_of_listnat listNatbase


(*
  final (infix) functions for adding, subtracting, multiplying, and dividing
  which take strings as arguments and return a string
*)
let (+++) n m =
 string_of_nat ((nat_of_string n) ++ (nat_of_string m))

let (---) n m =
 string_of_nat ((nat_of_string n) -- (nat_of_string m))

let (+**) n m =
 string_of_nat ((nat_of_string n) +* (nat_of_string m))

let (///) n m =
 string_of_nat ((nat_of_string n) // (nat_of_string m))

let (%%) n m =
string_of_nat ((nat_of_string n) % (nat_of_string m))
```

So putting it all together, we have a working calculator for natural numbers!

Let's try it out:
- `"3" +++ "17"` returns `"20"`
- `"182" --- "93"` returns `"89"`
- `"12" +** "3"` returns `"36"`
- `"41" /// "3"` returns `"13"`
- `"41" %% "3"` returns `"2"`

## Conclusion

We have built up natural numbers from first principles and now have a working calculator. However these operators start getting very slow for numbers of around 7 digits or more, so sticking with built-in integers sounds preferable..
