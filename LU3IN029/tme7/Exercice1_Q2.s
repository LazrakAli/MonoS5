.data
ch1:	.asciiz	"l exemple d'exemple\n"
	.align	1
ch2:	.asciiz	"HELLO WORLD\n"

.text
main:
# printf("%s", ch1)
	lui	$8,0x1001
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
# min_to_maj_chaine(ch1);
	lui	$8,0x1001
	ori	$4,$8,0
	jal	min_to_maj_chaine

# printf("%s", ch1)
	lui	$8,0x1001
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
	
# printf("%s", ch2)

	lui	$8,0x1001
	addiu	$8,$8,22
	ori	$4,$8,0
	ori	$2,$0,4
	syscall

	lui	$8,0x1001
	addiu	$8,$8,22
	ori	$4,$8,0
	jal	min_to_maj_chaine
# printf("%s", ch1)
	lui	$8,0x1001
	addiu	$8,$8,22
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
#exit()
	ori	$2,$0,10
	syscall

min_to_maj_chaine:
	addiu	$29,$29,-4
	sw	$31,0($29)
	ori	$15,$0,0x20
	lw	$14,0($29)
while:
#chargement chi
	lb	$9,0($4)
	
# ch[i] != ’\0’
	beq	$9,$0,endwhile

# if (ch[i] >= ’a’ && ch[i] <= ’z’)
	ori	$12,$0,0x61
    	slt     $12, $9, $12
    	bgtz    $12, endif
		
	ori	$12,$0,0x7A
    	slt     $12, $12,$9
    	bgtz    $12, endif
#ch[i] = ch[i] - 0x20;
	sub	$9,$9,$15
	sb	$9,0($4)

endif:
	addiu	$14,$14,1
	addiu	$4,$4,1
	j while

endwhile:
	addiu	$29,$29,4
	jr	$31
	
	