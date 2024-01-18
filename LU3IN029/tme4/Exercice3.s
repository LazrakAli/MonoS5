.data 

ch:	.asciiz	"coucou"

.text

	lui	$8,0x1001
#changement des 2 premiers char
	lb $9, 0($8)    # Charger le premier caractère dans $9
	lb $10, 1($8)   # Charger le deuxième caractère dans $10
	sb $10, 0($8)   # Stocker le deuxième caractère à la première position
	sb $9, 1($8)	# Stocker le premier caractère à la deuxième position
	
#affichage de ch	
	ori	$4, $8, 0
	ori	$2, $0, 4
	syscall
	
#exit 
	ori $2,	$0, 10
	syscall