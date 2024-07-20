from __future__ import annotations
from typing import Any, Optional
from threading import Lock


class LLNode:
    def __init__(self, value: Any):
        self.value: Any = value
        self.next: LLNode | None = None


class LinkedList:
    def __init__(self) -> None:
        self.head: Optional[LLNode] = None
        self.tail: Optional[LLNode] = None
        self._size: int = 0
        self._mu = Lock()

    def push_back(self, value: int) -> None:
        """Add an element to the end of the list."""
        new_node = LLNode(value)
        with self._mu:
            if not self.head:
                self.head = new_node
                self.tail = new_node
            else:
                self.tail.next = new_node
                self.tail = new_node
            self._size += 1

    def push_front(self, value: Any) -> None:
        """Add an element to the beginning of the list."""
        new_node = LLNode(value)
        with self._mu:
            if not self.head:
                self.head = new_node
                self.tail = new_node
            else:
                new_node.next = self.head
                self.head = new_node
            self._size += 1

    def pop_back(self) -> Any:
        """Remove the last element from the list and return its value."""
        with self._mu:
            if not self.head:
                raise IndexError("pop from empty list")
            removed_value = self.tail.value
            if self.head == self.tail:
                self.head = None
                self.tail = None
            else:
                current = self.head
                while current.next != self.tail:
                    current = current.next
                current.next = None
                self.tail = current
            self._size -= 1
        return removed_value

    def pop_front(self) -> Any:
        """Remove the first element from the list and return its value."""
        with self._mu:
            if not self.head:
                raise IndexError("pop from empty list")
            removed_value = self.head.value
            self.head = self.head.next
            if not self.head:
                self.tail = None
            self._size -= 1
        return removed_value

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        with self._mu:
            current = self.head
            while current:
                yield current.value
                with self._mu:
                    current = current.next

    def __repr__(self) -> str:
        return "LinkedList([" + ", ".join(map(str, self)) + "])"
