#include <stdio.h>
#include <stdlib.h>

/* Include the file list.h located in the current directory. */
#include "list.h"

/******************************
 * The public interface.  
 */

LIST *new_list(const char *value)
{
  // Create a new list, initialize its single node to "value",
  // and return a pointer to the list.
LIST *n_list = malloc(sizeof(LIST));
NODE *new_node = make_node(value); 
n_list ->head = new_node;
n_list ->tail = new_node;
return n_list;

}

void prepend(LIST* const list, const char* const value)
{
  /* Add a new node at the head of the list. */
  /* Update the head pointer in the list. */
  NODE *new_node = make_node(value);
  new_node ->next = list -> head;
  list -> head -> previous = new_node;
  list -> head = new_node;
}

void append(LIST* const list, const char* const value)
{
  /* Add a new node at the tail of the list. */
  /* Update the tail pointer in the list. */  
}

void delete_list(LIST *list)
{
  /* Delete a list and free its allocated memory. */
}

/*********************************************************
 * The following code is provided for your convenience.
 * You do not have to use it.
 */

void print_list(const LIST* const list)
{
  /* Print the contents of a list. */
  for (NODE *node = list->head; node != NULL; node = node->next) {
    printf("%s\n", node->value);
  }
}

static NODE *make_node(const char *value) 
{
  NODE *new_node = malloc(sizeof(NODE));
  new_node->value = strdup(value);
  new_node->previous = NULL;
  new_node->next = NULL;
  return new_node;
}